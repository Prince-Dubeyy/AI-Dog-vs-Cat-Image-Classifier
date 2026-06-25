from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dog vs Cat Classifier API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "dog_cat_classifier.keras")
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess matching training:
        # - RGB conversion
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # - Resize to 128x128
        image = image.resize((128, 128))
        
        # - Convert to numpy array (float32)
        img_array = np.array(image, dtype=np.float32)
        
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
            
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
