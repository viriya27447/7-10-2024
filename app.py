import streamlit as st
import numpy as np
import requests
import tempfile
import os
from keras.models import load_model
from PIL import Image, ImageOps
import io

# URL ของไฟล์โมเดลและไฟล์ labels ที่อยู่บน GitHub
model_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee.h5?alt=media&token=5f2aa892-3780-429f-96a3-c47ac9fbf689"  # เปลี่ยนเป็น URL ของโมเดล
labels_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee-labels.txt?alt=media&token=7b5cd9d4-9c27-4008-a58d-5b0db0acd8f4"  # เปลี่ยนเป็น URL ของ labels

# ดาวน์โหลดโมเดล
response = requests.get(model_url)
model_file = io.BytesIO(response.content)

# บันทึกไฟล์โมเดลชั่วคราว
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
    temp_file.write(model_file.getbuffer())
    temp_model_path = temp_file.name

# โหลดโมเดล
model = load_model(temp_model_path, compile=False)

# ดาวน์โหลด labels
response = requests.get(labels_url)
class_names = response.text.splitlines()

# สร้างอาร์เรย์ที่มีขนาดที่ถูกต้องสำหรับโมเดล Keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# สร้างคอลัมน์สำหรับอินพุตและเอาท์พุต
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    option = st.selectbox("Choose input method", ("Upload Image", "Open Camera"))

    image = None  # Initialize image variable

    if option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption='Uploaded Image', use_column_width=True)

    elif option == "Open Camera":
        uploaded_image = st.camera_input("Take a picture")

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption='Captured Image', use_column_width=True)

with col2:
    st.header("Output")

    if image is not None:
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.BICUBIC)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.write("Class:", class_name[2:])
        st.write("Confidence Score:", confidence_score)
    else:
        st.warning("Please upload an image or capture one to proceed.")

# ลบไฟล์ชั่วคราวเมื่อเสร็จสิ้น
os.remove(temp_model_path)