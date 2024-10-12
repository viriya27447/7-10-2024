import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os

# Function to validate URL
def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except Exception:
        return False

def page1():
    st.write("Welcome to the Coffee Classifier App! Please navigate to Page 2 to classify coffee images.")

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

# ใช้ st.sidebar เพื่อให้เลือกหน้าได้
page = st.sidebar.radio("Select Page", ["Page 1", "Page 2"])

if page == "Page 1":
    page1()
else:
    page2()
