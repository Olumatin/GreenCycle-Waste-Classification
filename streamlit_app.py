import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import json
from huggingface_hub import hf_hub_download


# Page configuration
st.set_page_config(
    page_title="GreenCycle Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #1b5e20;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    '<div class="main-title">♻️ GreenCycle Waste Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image of waste and let the AI classify it.</div>',
    unsafe_allow_html=True
)


# Load model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Olumatin/greencycle-waste-classifier",
        filename="waste_classifier.keras"
    )

    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()


# File uploader
uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Waste", use_container_width=True):

        with st.spinner("Analyzing image..."):

            resized_image = image.resize((256, 256))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

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

        # Display result
        st.markdown(
            f"""
            <div class="result-box">
                <h2>Prediction</h2>
                <h1>{predicted_class.title()}</h1>
                <p><strong>Confidence:</strong>
                {confidence * 100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info(
        "Please upload a JPG, JPEG, or PNG image to begin."
    )


# Supported waste categories
st.markdown("---")

st.subheader("Supported Waste Categories")

st.write(
    ", ".join(
        category.title()
        for category in class_names
    )
)

st.markdown("---")

st.caption(
    "GreenCycle Waste Classification System | "
    "CNN-based image classification"
)
