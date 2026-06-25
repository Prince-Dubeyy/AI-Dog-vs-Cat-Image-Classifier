import tensorflow as tf
import numpy as np
import requests
import io
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def get_random_cat_url():
    try:
        response = requests.get('https://api.thecatapi.com/v1/images/search')
        return response.json()[0]['url']
    except:
        return None

def get_random_dog_url():
    try:
        response = requests.get('https://dog.ceo/api/breeds/image/random')
        return response.json()['message']
    except:
        return None

def process_image(url, model):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(io.BytesIO(response.content))
        
        # Preprocess exactly as in main.py
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction_prob = model.predict(img_array, verbose=0)[0][0]
        
        if prediction_prob > 0.5:
            prediction = "Dog"
            confidence = float(prediction_prob) * 100
        else:
            prediction = "Cat"
            confidence = float(1 - prediction_prob) * 100
            
        return prediction, confidence
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None

def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "dog_cat_classifier.keras"))
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    y_true = []
    y_pred = []
    
    print("\n--- Testing 10 Cat Images ---")
    cats_tested = 0
    while cats_tested < 10:
        url = get_random_cat_url()
        if not url: continue
        pred, conf = process_image(url, model)
        if pred:
            print(f"Image: {url[:50]}... | Actual: Cat | Predicted: {pred} | Confidence: {conf:.2f}%")
            y_true.append("Cat")
            y_pred.append(pred)
            cats_tested += 1

    print("\n--- Testing 10 Dog Images ---")
    dogs_tested = 0
    while dogs_tested < 10:
        url = get_random_dog_url()
        if not url: continue
        pred, conf = process_image(url, model)
        if pred:
            print(f"Image: {url[:50]}... | Actual: Dog | Predicted: {pred} | Confidence: {conf:.2f}%")
            y_true.append("Dog")
            y_pred.append(pred)
            dogs_tested += 1

    print("\n==================================================")
    print("PHASE 5 — MODEL VALIDATION RESULTS")
    print("==================================================")
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
    
    print("Confusion Matrix (Cat=0, Dog=1):")
    print(confusion_matrix(y_true, y_pred, labels=["Cat", "Dog"]))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["Cat", "Dog"]))

if __name__ == "__main__":
    main()
