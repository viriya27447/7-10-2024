import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the model while skipping 'groups' in DepthwiseConv2D
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove 'groups' if present in kwargs
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

# Load the model from the URL
@st.cache_resource
def load_custom_model():
    model_url = "https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/coffee.h5?alt=media&token=5f2aa892-3780-429f-96a3-c47ac9fbf689"
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
    count_file = "visitor_count.csv"
    
    # Check if the count file exists; if not, create it
    if not os.path.exists(count_file):
        with open(count_file, 'w') as f:
            f.write("timestamp,count\n")  # Write header
    
    # Read the current count
    df = pd.read_csv(count_file)
    
    # Get current count
    count = len(df)  # Count is the number of rows (visitors)
    
    # Add new visitor
    new_data = pd.DataFrame({"timestamp": [pd.Timestamp.now()], "count": [count + 1]})
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(count_file, index=False)  # Save updated data

    return count, df

# Streamlit app section
st.markdown("<h1 style='text-align: center;'>Coffee Classifier</h1>", unsafe_allow_html=True)

# Update visitor count
visitor_count, visitor_data = update_visitor_count()
st.sidebar.write(f"Total Visitor Count: {visitor_count}")

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

# Plot visitor count over time
st.subheader("Visitor Count Over Time")
plt.figure(figsize=(10, 5))
plt.plot(visitor_data['timestamp'], visitor_data['count'], marker='o', linestyle='-', color='b')
plt.title("Visitor Count Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Total Visitors")
plt.xticks(rotation=45)
plt.grid()
st.pyplot(plt)
