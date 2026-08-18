
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io

app = FastAPI(
    title="GreenCycle Waste Classification API",
    description="CNN-based waste classification API",
    version="1.0"
)

# Load trained model
model = tf.keras.models.load_model("waste_classifier.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.get("/")
def home():
    return {
        "message": "GreenCycle Waste Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize image to model input size
    image = image.resize((256, 256))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    }
    
