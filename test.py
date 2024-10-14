import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load embeddings and filenames from pickle files
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a model by adding GlobalMaxPooling2D layer on top of ResNet50
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load and preprocess the query image
img = image.load_img('samples/2024-04-28 (1).png', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features from the query image
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Find nearest neighbors in the feature space
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

# Print the filenames of the nearest neighbors
print(indices)
for file_index in indices[0]:
    if 0 <= file_index < len(filenames):
        print(filenames[file_index])
