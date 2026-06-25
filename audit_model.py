import tensorflow as tf
import os

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "dog_cat_classifier.keras"))
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)
model.summary()
