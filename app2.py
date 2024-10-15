import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os

def page1():
    
    def custom_depthwise_conv2d(*parameters, **keyword_parameters):
    # ลบคีย์ 'groups' ออกจาก keyword_parameters หากมีการส่งเข้ามา
        keyword_parameters.pop('groups', None)
    
    # เรียกใช้เลเยอร์ DepthwiseConv2D จาก TensorFlow ด้วยพารามิเตอร์ที่เหลือ
        return tf.keras.layers.DepthwiseConv2D(*parameters, **keyword_parameters)

    # Load the model from the embedded URL
    @st.cache_resource
    def load_custom_model():
        model_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/12102024.h5?alt=media&token=8cdfbf0a-4ec6-4e59-bd35-d420890f8166"
        temp_model_path = os.path.join(tempfile.gettempdir(), 'coffee_model.h5')

        response = requests.get(model_url)
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)

        model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
        return model

    def load_labels():
        labels_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee-labels.txt?alt=media&token=7b5cd9d4-9c27-4008-a58d-5b0db0acd8f4"
        response = requests.get(labels_url)
        class_names = response.text.splitlines()
        return class_names

    def predict(image, model, class_names):
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        return prediction

    st.markdown("<h1 style='text-align: center;'>Coffee Classifier</h1>", unsafe_allow_html=True)
    
    model = load_custom_model()
    class_names = load_labels()

    col1, col2 = st.columns(2)
    with col1:
        # Toggle between uploading an image and taking a picture
        mode = st.radio("Select Mode", ["Upload Image", "Take a Picture"])

        uploaded_file = None
        camera_file = None
        class_name = ""
        confidence_score = 0.0

        if mode == "Upload Image":
            # Upload image (supports both PNG and JPG)
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

                st.success("Image uploaded successfully!")
            else:
                st.warning("Please upload an image to proceed.")

        else:
            camera_file = st.camera_input("Take a picture")
            if camera_file is not None:
                image = Image.open(camera_file)
                st.image(image, caption='Captured Image.', use_column_width=True)

                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

                st.success("Picture captured successfully!")
            else:
                st.warning("Please take a picture to proceed.")

    with col2:
        st.header("Prediction Result")
        if (mode == "Upload Image" and uploaded_file is not None) or (mode == "Take a Picture" and camera_file is not None):
            st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
            st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
        else:
            st.write("Please upload an image or take a picture to see the prediction.")

def page2():
    st.write('hello test')

pg = st.navigation([st.Page(page1), st.Page(page2)])
pg.run()