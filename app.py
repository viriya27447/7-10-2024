import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os

# ฟังก์ชันสำหรับโหลดโมเดลโดยข้าม 'groups' ใน DepthwiseConv2D
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # ลบ 'groups' ถ้ามีใน kwargs
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

# โหลดโมเดลจาก URL
@st.cache_resource
def load_custom_model():
    model_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee.h5?alt=media&token=5f2aa892-3780-429f-96a3-c47ac9fbf689"
    temp_model_path = os.path.join(tempfile.gettempdir(), 'coffee_model.h5')

    # ดาวน์โหลดไฟล์โมเดลจาก URL
    response = requests.get(model_url)
    with open(temp_model_path, 'wb') as f:
        f.write(response.content)

    # โหลดโมเดลโดยใช้ custom_objects
    model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
    return model

# โหลด labels จาก URL
def load_labels():
    labels_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee-labels.txt?alt=media&token=7b5cd9d4-9c27-4008-a58d-5b0db0acd8f4"
    response = requests.get(labels_url)
    class_names = response.text.splitlines()
    return class_names

# ฟังก์ชันสำหรับทำนายผล
def predict(image, model, class_names):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    return prediction

# ส่วนของ Streamlit app
st.title("Coffee Classifier")

# โหลดโมเดลและ labels
model = load_custom_model()
class_names = load_labels()

# สวิตช์ระหว่างการอัปโหลดรูปภาพและการถ่ายภาพสด
mode = st.toggle("Select Mode", ["อัปโหลดรูป", "ถ่ายรูป"])

if mode == "อัปโหลดรูป":
    # อัปโหลดรูปภาพ (รองรับทั้ง PNG และ JPG)
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # ทำนายผล
        prediction = predict(image, model, class_names)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence_score * 100:.2f}%")  # แสดงเป็นเปอร์เซ็นต์

else:
    # ถ่ายภาพจากกล้อง
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)
        st.image(image, caption='Captured Image.', use_column_width=True)

        # ทำนายผล
        prediction = predict(image, model, class_names)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence_score * 100:.2f}%")  # แสดงเป็นเปอร์เซ็นต์
