import streamlit as st
import numpy as np
import zipfile
from keras.models import load_model
from PIL import Image, ImageOps
from zipfile import ZipFile

# กำหนดเส้นทางไฟล์ ZIP ของโมเดล
uploaded_model = r"C:\Users\acer\OneDrive\เดสก์ท็อป\ป.ตรี\project-AI[6-10-2024]\converted_keras.zip"

# เปิดไฟล์ ZIP ในโหมดอ่าน
with ZipFile(uploaded_model, 'r') as zip:
    zip.extractall()  # แตกไฟล์ไปยังไดเรกทอรีปัจจุบัน

# ปิดการแสดงผลเลขทศนิยมวิทยาศาสตร์เพื่อความชัดเจน
np.set_printoptions(suppress=True)

# โหลดโมเดล
model = load_model("D:/converted_keras/coffee.h5", compile=False)

# โหลด labels
class_names = open("D:/converted_keras/coffee-labels.txt", "r").readlines()

# สร้างอาร์เรย์ที่มีขนาดที่ถูกต้องสำหรับโมเดล Keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# สร้างคอลัมน์สำหรับอินพุตและเอาท์พุต
col1, col2 = st.columns(2)

with col1:
    st.header("นำรูปเมล็ดกาแฟเข้า")
    # ตัวเลือกในการอัปโหลดรูปภาพหรือเปิดกล้อง
    option = st.selectbox("Choose input method", ("Upload Image", "Open Camera"))

    if option == "Upload Image":
        # อัปโหลดไฟล์ภาพ (รองรับทั้ง jpg และ png)
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

        # ตรวจสอบว่าผู้ใช้ได้อัปโหลดไฟล์หรือไม่
        if uploaded_image is not None:
            # เปิดและแปลงรูปภาพ
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption='Uploaded Image', use_column_width=True)

    elif option == "Open Camera":
        # ใช้ฟังก์ชัน st.camera_input เพื่อเปิดกล้อง
        uploaded_image = st.camera_input("Take a picture")

        # ตรวจสอบว่าผู้ใช้ได้ถ่ายภาพหรือไม่
        if uploaded_image is not None:
            # เปิดและแปลงรูปภาพ
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption='Captured Image', use_column_width=True)

with col2:
    st.header("ผลการทำนาย")
    
    if 'image' in locals():  # ตรวจสอบว่ามีภาพที่ต้องประมวลผลหรือไม่
        # ปรับขนาดรูปภาพให้เป็นอย่างน้อย 224x224 และครอบตัดจากกลาง
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.BICUBIC)

        # แปลงรูปภาพเป็นอาร์เรย์ NumPy
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # .strip() removes any extra whitespace
        confidence_score = prediction[0][index]

        # แสดงผลการทำนายและคะแนนความมั่นใจใน Streamlit
        st.write("Class:", class_name[2:])  # ตัดสองตัวแรกของชื่อคลาส
        st.write("Confidence Score:", confidence_score)
    else:
        st.warning("Please upload an image or capture one to proceed.")