import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===============================
# Load Model (.keras ONLY)
# ===============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.keras")
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
st.write("Upload gambar (28x28 grayscale) lalu klik Prediksi")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Preview image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=150)

    # Tombol prediksi
    if st.button("🔍 Prediksi"):
        # ===============================
        # Preprocessing
        # ===============================
        image_resized = image.resize((28, 28))
        img_array = np.array(image_resized)

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