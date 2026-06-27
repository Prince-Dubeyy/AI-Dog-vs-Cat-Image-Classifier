import os
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
ood_model = None

# TASK 3: Robust CORS parsing from environment
raw_frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173,http://localhost:3000")
allowed_origins = [
    origin.strip().rstrip('/') # Also strip trailing slashes defensively
    for origin in raw_frontend_url.split(",")
    if origin.strip()
]

# TASK 4: Add temporary debugging during startup
logger.info(f"Loaded FRONTEND_URL: {raw_frontend_url}")
logger.info(f"Loaded allowed_origins: {allowed_origins}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ood_model
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "dog_cat_classifier.keras"))
    try:
        logger.info(f"Application starting up... Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        
    try:
        logger.info("Loading MobileNetV2 for OOD detection...")
        ood_model = tf.keras.applications.MobileNetV2(weights='imagenet')
        logger.info("OOD model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load OOD model: {e}")
        ood_model = None

    yield
    logger.info("Application shutting down...")
    model = None
    ood_model = None

app = FastAPI(title="Dog vs Cat Classifier API", lifespan=lifespan)

# TASK 8: Verify middleware ordering (CORS added first, executes outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# TASK 7: Global exception handlers to inject CORS headers manually if needed
# Note: CORSMiddleware usually handles this, but explicit handlers guarantee it
def get_cors_headers(request: Request):
    origin = request.headers.get("origin")
    if origin and (origin in allowed_origins or "*" in allowed_origins):
        return {"Access-Control-Allow-Origin": origin}
    return {}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    headers = get_cors_headers(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    headers = get_cors_headers(request)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers=headers
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    headers = get_cors_headers(request)
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the request."},
        headers=headers
    )

# TASK 5: Add /debug/cors endpoint
@app.get("/debug/cors")
async def debug_cors():
    return {
        "allowed_origins": allowed_origins,
        "frontend_env": raw_frontend_url,
        "environment": "production"
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Dog vs Cat Classifier API. Use /docs to view the API documentation."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        logger.error("Prediction request failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="File provided is not an image")

    try:
        # Read image
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File size exceeded: {len(contents)} bytes")
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

        try:
            image = Image.open(io.BytesIO(contents))
            # Verify the image is valid by loading it
            image.verify()
            # Image.verify() closes the file, need to reopen
            image = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            logger.warning("Uploaded file is corrupted or not a valid image format.")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")
        
        logger.info(f"Received prediction request for file: {file.filename}")

        # Preprocess matching training exactly:
        # - RGB conversion
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array first
        img_array = np.array(image)
        
        # - Resize using tf.image.resize to match training exactly
        img_array = tf.image.resize(img_array, (128, 128))
        
        # - Convert to float32
        img_array = tf.cast(img_array, tf.float32)
        
        # - Preprocess for MobileNetV2 (scales pixels to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # - Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # --- OOD Detection (Check if image is likely a dog or cat) ---
        if ood_model is not None:
            # MobileNetV2 expects 224x224
            ood_img = tf.image.resize(np.array(image), (224, 224))
            ood_img = tf.keras.applications.mobilenet_v2.preprocess_input(ood_img)
            ood_img = np.expand_dims(ood_img, axis=0)
            ood_preds = ood_model.predict(ood_img)[0]
            
            # Check top 3 predictions
            top_3_classes = np.argsort(ood_preds)[-3:][::-1]
            found_pet = False
            for cls in top_3_classes:
                # ImageNet dog classes: ~151 to 268. Cat classes: ~281 to 285.
                if (151 <= cls <= 268) or (281 <= cls <= 285):
                    found_pet = True
                    break
            
            if not found_pet:
                logger.info(f"OOD detected for {file.filename}. Top classes: {top_3_classes}")
                return JSONResponse(content={
                    "prediction": "Neither",
                    "confidence": 0.0
                })
        
        # Predict
        prediction_prob = model.predict(img_array)[0][0]
        
        # Output logic
        # Dog if > 0.5, else Cat
        if prediction_prob > 0.5:
            prediction = "Dog"
            confidence = float(prediction_prob) * 100
        else:
            prediction = "Cat"
            confidence = float(1 - prediction_prob) * 100
            
        logger.info(f"Prediction successful for {file.filename}: {prediction} ({confidence:.2f}%)")
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled exception during prediction.")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the image.")
