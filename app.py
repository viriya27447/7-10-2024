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

# Streamlit app section
st.title("Coffee Classifier")

# Load model and labels
model = load_custom_model()
class_names = load_labels()

# Dropdown for selecting an example
example = st.selectbox("Select an example:", ["Example 1", "Example 2", "Example 3"])

# Creating a table based on the selected example
if example == "Example 1":
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
elif example == "Example 2":
    data = np.array([[10, 11, 12],
                     [13, 14, 15],
                     [16, 17, 18]])
elif example == "Example 3":
    data = np.array([[19, 20, 21],
                     [22, 23, 24],
                     [25, 26, 27]])

# Display the table
st.write("### Table:")
st.dataframe(data)

# Rest of your Streamlit application code can go here
