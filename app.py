import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ==============================
# LOAD MODEL (SAVED_MODEL)
# ==============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("saved_model")
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model = load_model()

labels = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==============================
# UI
# ==============================
st.title("👕 Fashion MNIST Classifier")
st.info("Upload gambar lalu lihat hasil prediksi")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess(image):
    image = image.convert('L')

    # Resize + crop (biar tidak distorsi)
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)

    img_array = np.array(image).astype("float32") / 255.0

    # Auto invert (biar cocok dengan dataset)
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    # Normalisasi kontras
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)

    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, image

# ==============================
# PREDICT
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    if model is None:
        st.error("Model gagal dimuat. Periksa folder saved_model")
    else:
        img_array, processed = preprocess(image)

        st.image(processed, caption="Processed (Input Model)", width=200)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: {labels[class_index]}")
        st.write(f"Confidence: {confidence:.4f}")

        # ==============================
        # TOP 3
        # ==============================
        st.subheader("Top 3 Predictions")
        top3 = np.argsort(prediction[0])[-3:][::-1]

        for i in top3:
            st.write(f"{labels[i]}: {prediction[0][i]:.4f}")