import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import tensorflow as tf  # Import TensorFlow as tf


feature_list = None
filenames = None

try:
    feature_list = pickle.load(open('embeddings.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except FileNotFoundError:
    st.error("Pickle files not found. Make sure they exist in the specified locations.")
except pickle.UnpicklingError as e:
    st.error(f"An error occurred while loading pickle files: {e}. This might be due to corrupted or incomplete data.")
except Exception as e:
    st.error(f"An error occurred while loading pickle files: {e}")

# Load pre-trained ResNet50 model
model = tf.keras.Sequential([  # Use tf.keras.Sequential
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalMaxPooling2D()
])

st.title('Style-R')
st.subheader('Welcome to Style-R. Discover your own style')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from an image
def feature_extraction(image_path, model):
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print("Error loading or processing image:", e)
        return None


# Function to recommend similar images
def recommend(features, feature_list, filenames):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)



        # Recommend similar images
        indices = recommend(features, feature_list, filenames)

        # Display recommended images
        if len(indices) > 0:
            st.header("Recommended Images:")
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.image(filenames[indices[0][i]], use_column_width=True)
        else:
            st.error("No recommendations found.")
