import os
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "dog_cat_classifier.keras"))
    try:
        logger.info(f"Application starting up... Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
    yield
    logger.info("Application shutting down...")
    model = None

app = FastAPI(title="Dog vs Cat Classifier API", lifespan=lifespan)

# CORS Configuration
# Allow localhost for development and an environment variable for production
ALLOWED_ORIGINS = os.getenv("FRONTEND_URL", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
