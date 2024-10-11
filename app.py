import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os

# Function to validate URLs
def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except Exception:
        return False

def page1():
    # Function to load the model while skipping 'groups' in DepthwiseConv2D
    def custom_depthwise_conv2d(*args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if present in kwargs
        return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

    # Load the model from the URL
    @st.cache_resource
    def load_custom_model():
        model_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/12102024.h5?alt=media&token=8cdfbf0a-4ec6-4e59-bd35-d420890f8166"
        temp_model_path = os.path.join(tempfile.gettempdir(), 'coffee_model.h5')

        # Download the model file from the URL
        response = requests.get(model_url)
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)

        # Load the model using custom_objects
        model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
        return model

    # Load labels from the URL
    def load_labels():
        labels_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee-labels.txt?alt=media&token=7b5cd9d4-9c27-4008-a58d-5b0db0acd8f4"
        response = requests.get(labels_url)
        class_names = response.text.splitlines()
        return class_names

    # Function to make predictions
    def predict(image, model, class_names):
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        return prediction

    # Function to read and update visitor count
    def update_visitor_count():
        count_file = "visitor_count.txt"

        # Check if the count file exists; if not, create it and initialize to 0
        if not os.path.exists(count_file):
            with open(count_file, 'w') as f:
                f.write("0")

        # Read the current count
        with open(count_file, 'r') as f:
            count = int(f.read().strip())

        # Increment the count and write back to the file
        count += 1
        with open(count_file, 'w') as f:
            f.write(str(count))

        return count

    # Streamlit app section
    st.markdown("<h1 style='text-align: center;'>Coffee Classifier</h1>", unsafe_allow_html=True)

    # Load model and labels
    model = load_custom_model()
    class_names = load_labels()

    # Create columns for input and output
    col1, col2 = st.columns(2)

    with col1:
        # Toggle between uploading an image and taking a picture
        mode = st.radio("Select Mode", ["Upload Image", "Take a Picture"])

        if mode == "Upload Image":
            # Upload image (supports both PNG and JPG)
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                # Make predictions
                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

        else:
            # Take a picture from the camera
            camera_file = st.camera_input("Take a picture")
            if camera_file is not None:
                image = Image.open(camera_file)
                st.image(image, caption='Captured Image.', use_column_width=True)

                # Make predictions
                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

    with col2:
        # This section is for displaying the prediction result
        st.header("Prediction Result")
        if mode == "Upload Image" and uploaded_file is not None:
            st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
            st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
        elif mode == "Take a Picture" and camera_file is not None:
            st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
            st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
        else:
            st.write("Please upload an image or take a picture to see the prediction.")

    def display_image_table():
        # ข้อมูลสำหรับตาราง (2x2)
        table_data = [
            ["https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee%20exemple%20img%2Fdark%20(1).png?alt=media&token=5d626d79-7203-43f9-9a14-345d94f20935", 
             "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee%20exemple%20img%2Fgreen%20(2).png?alt=media&token=a475026b-e69a-4713-b9a2-96d7fadfcb2b"],
            ["https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee%20exemple%20img%2Flight%20(1).png?alt=media&token=b87e27d4-0dfd-4746-a713-6ec2567d819d", 
             "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee%20exemple%20img%2Fmedium%20(1).png?alt=media&token=3f661e8a-bf6c-4061-9a6d-19bb9994c151"]
        ]
        
        # สร้างปุ่มสำหรับซ่อน/แสดงตาราง
        if st.button("Image Example"):
            # ใช้ session state เพื่อควบคุมการแสดงผลของตาราง
            if "show_table" not in st.session_state:
                st.session_state.show_table = True  # ตั้งค่าเริ่มต้นเป็น True
                
            # สลับสถานะการแสดงผล
            st.session_state.show_table = not st.session_state.show_table

        # แสดงตารางถ้าสถานะ show_table เป็น True
        if st.session_state.get("show_table", False):
            st.subheader("Image Table")

            # ใช้ container เพื่อจัดกลาง
            with st.container():
                # สร้างคอลัมน์เพื่อจัดรูปภาพกลาง
                cols = st.columns(len(table_data[0]))  # สร้างคอลัมน์ตามจำนวนของข้อมูลในแถว
                for row in table_data:
                    for col, item in zip(cols, row):
                        # ตรวจสอบว่าข้อมูลเป็น URL ของรูปภาพหรือไม่
                        if item.startswith("http"):
                            col.image(item, width=100)  # กำหนดขนาดรูปภาพเป็น 100 พิกเซล
                        else:
                            col.write(item)  # แสดงข้อความถ้าไม่ใช่รูปภาพ
                
                st.markdown("See More : [https://drive.google.com/drive/folders/AI/รูปกาแฟคั่วถ่ายเอง+kaggle](https://drive.google.com/drive/folders/13mdUTt9wMn-swYButWDfugoCFJoA-DHo?usp=drive_link)")
    
    display_image_table()

    st.write('Presented by : Group 5 Student ID 65050225,65050686,65050378,65050838')

def page2():
    # Function to load the model while skipping 'groups' in DepthwiseConv2D
    def custom_depthwise_conv2d(*args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if present in kwargs
        return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

    # Load the model from the URL
    @st.cache_resource
    def load_custom_model(model_url):
        temp_model_path = os.path.join(tempfile.gettempdir(), 'coffee_model.h5')

        # Download the model file from the URL
        response = requests.get(model_url)
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)

        # Load the model using custom_objects
        model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
        return model

    # Load labels from the URL
    def load_labels(labels_url):
        response = requests.get(labels_url)
        class_names = response.text.splitlines()
        return class_names

    # Function to make predictions
    def predict(image, model, class_names):
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        return prediction

    # Streamlit app section for page 2
    st.markdown("<h1 style='text-align: center;'>Coffee Classifier</h1>", unsafe_allow_html=True)

    # Input URLs for the model and labels
    model_url = st.text_input("Enter the model URL:", 
        "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/12102024.h5?alt=media&token=8cdfbf0a-4ec6-4e59-bd35-d420890f8166")
    
    labels_url = st.text_input("Enter the labels URL:", 
        "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee-labels.txt?alt=media&token=7b5cd9d4-9c27-4008-a58d-5b0db0acd8f4")

    # Validate URLs and load model/labels
    if st.button("Load Model and Labels"):
        if not is_valid_url(model_url):
            st.error("Invalid model URL. Please check the URL and try again.")
        elif not is_valid_url(labels_url):
            st.error("Invalid labels URL. Please check the URL and try again.")
        else:
            # Load model and labels
            st.session_state.model = load_custom_model(model_url)
            st.session_state.class_names = load_labels(labels_url)
            st.success("Model and labels loaded successfully!")

    # Check if the model and labels are loaded
    if 'model' in st.session_state and 'class_names' in st.session_state:
        model = st.session_state.model
        class_names = st.session_state.class_names
        
        # Create columns for input and output
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

                    # Make predictions
                    prediction = predict(image, model, class_names)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip()
                    confidence_score = prediction[0][index]

            else:
                # Take a picture from the camera
                camera_file = st.camera_input("Take a picture")
                if camera_file is not None:
                    image = Image.open(camera_file)
                    st.image(image, caption='Captured Image.', use_column_width=True)

                    # Make predictions
                    prediction = predict(image, model, class_names)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip()
                    confidence_score = prediction[0][index]

        with col2:
            # This section is for displaying the prediction result
            st.header("Prediction Result")
            if mode == "Upload Image" and uploaded_file is not None:
                st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
                st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
            elif mode == "Take a Picture" and camera_file is not None:
                st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
                st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
            else:
                st.write("Please upload an image or take a picture to see the prediction.")

    st.write('Presented by : Group 5 Student ID 65050225,65050686,65050378,65050838')

pg = st.navigation([st.Page(page1), st.Page(page2)])
pg.run()
