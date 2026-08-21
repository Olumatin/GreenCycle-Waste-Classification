from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os

from huggingface_hub import hf_hub_download


app = FastAPI(
    title="GreenCycle Waste Classification API",
    description="CNN-based waste classification API",
    version="1.0"
)


# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Download trained model from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="Olumatin/greencycle-waste-classifier",
    filename="waste_classifier.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# Load class names
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# User interface
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GreenCycle Waste Classifier</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f1f8f4;
                margin: 0;
                padding: 40px;
                text-align: center;
            }

            .container {
                max-width: 700px;
                margin: auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                color: #1b5e20;
            }

            p {
                color: #555;
                font-size: 18px;
            }

            input {
                margin: 20px;
                padding: 10px;
            }

            button {
                background: #2e7d32;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #1b5e20;
            }

            #result {
                margin-top: 25px;
                padding: 20px;
                background: #e8f5e9;
                border-radius: 10px;
                display: none;
            }

            #preview {
                max-width: 300px;
                margin: 20px auto;
                display: none;
                border-radius: 10px;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>♻️ GreenCycle Waste Classifier</h1>

            <p>
                Upload an image of waste and let the AI classify it.
            </p>

            <input
                type="file"
                id="fileInput"
                accept="image/png,image/jpeg"
            >

            <br>

            <img id="preview">

            <br>

            <button onclick="classifyWaste()">
                🔍 Classify Waste
            </button>

            <div id="result"></div>

        </div>


        <script>

        const fileInput = document.getElementById("fileInput");
        const preview = document.getElementById("preview");
        const result = document.getElementById("result");


        fileInput.addEventListener("change", function() {

            const file = fileInput.files[0];

            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = "block";
            }

        });


        async function classifyWaste() {

            const file = fileInput.files[0];

            if (!file) {
                alert("Please select a waste image first.");
                return;
            }

            result.style.display = "block";

            result.innerHTML = "🔄 Analyzing image...";


            const formData = new FormData();

            formData.append("file", file);


            try {

                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });


                const data = await response.json();


                if (!response.ok) {
                    throw new Error(
                        data.detail || "Prediction failed"
                    );
                }


                const confidence =
                    (data.confidence * 100).toFixed(2);


                result.innerHTML = `
                    <h2>Prediction</h2>
                    <h1>${data.predicted_class}</h1>
                    <p>
                        <strong>Confidence:</strong>
                        ${confidence}%
                    </p>
                `;


            } catch (error) {

                result.innerHTML =
                    "❌ Error: " + error.message;

            }

        }

        </script>

    </body>
    </html>
    """


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # Resize image
    image = image.resize((256, 256))


    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )


    # Normalize pixel values
    image_array = image_array / 255.0


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )


    predicted_index = int(
        np.argmax(predictions[0])
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = float(
        predictions[0][predicted_index]
    )


    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    }
