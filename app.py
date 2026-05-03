import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ===============================
# Load Model (Robust Version)
# ===============================
@st.cache_resource
def load_model():
    model_path = None

    # Prioritas pakai format baru (.keras)
    if os.path.exists("model.keras"):
        model_path = "model.keras"
    elif os.path.exists("model.h5"):
        model_path = "model.h5"
    else:
        st.error("Model tidak ditemukan!")
        return None

    try:
        # Load dengan safe mode
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        return model

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None


model = load_model()

# ===============================
# Label Fashion MNIST
# ===============================
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ===============================
# UI
# ===============================
st.title("👕 Fashion MNIST Classifier")
st.write("Upload gambar (28x28 grayscale) untuk prediksi")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=150)

    # ===============================
    # Preprocessing
    # ===============================
    image = image.resize((28, 28))
    img_array = np.array(image)

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # ===============================
    # Predict
    # ===============================
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # ===============================
    # Output
    # ===============================
    st.success(f"Prediksi: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {confidence:.2f}")