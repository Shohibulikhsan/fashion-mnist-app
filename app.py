import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================

# CONFIG PAGE

# =========================

st.set_page_config(
page_title="Fashion Classifier",
page_icon="👕",
layout="centered"
)

# =========================

# LOAD MODEL (CACHE BIAR CEPAT)

# =========================

@st.cache_resource
def load_model():
return tf.keras.models.load_model("model.h5")

model = load_model()

# =========================

# LABEL

# =========================

labels = [
"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =========================

# PREPROCESSING

# =========================

def preprocess_image(image):
image = image.convert("L")            # RGB → Grayscale
image = image.resize((28, 28))        # Resize
image = np.array(image) / 255.0       # Normalisasi
image = image.reshape(1, 28, 28, 1)   # Reshape
return image

# =========================

# UI

# =========================

st.title("👕 Fashion MNIST Classifier")
st.write("Upload gambar pakaian (sepatu, baju, dll) untuk diprediksi oleh model CNN")

# Sidebar

st.sidebar.title("Tentang")
st.sidebar.info(
"Aplikasi ini menggunakan model CNN yang dilatih pada dataset Fashion MNIST.\n\n"
"Model melakukan klasifikasi gambar ke dalam 10 kategori pakaian."
)

# Upload file

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
image = Image.open(uploaded_file)

```
st.image(image, caption="Gambar Input", use_column_width=True)

try:
    # Preprocess
    img = preprocess_image(image)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Output
    st.subheader("Hasil Prediksi:")
    st.success(f"{labels[predicted_class]}")

    st.write(f"Confidence: {confidence:.2f}")

    # Probabilitas
    st.subheader("Distribusi Probabilitas:")
    st.bar_chart(prediction[0])

except Exception as e:
    st.error(f"Terjadi error: {e}")
```
