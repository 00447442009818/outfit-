import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from  tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_feature(img_path,model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result /  norm(result)
        return normalized_result
    except:
        print("Error processing image:", img_path)
        return None

images_dir = os.path.join('images')
filenames = []
feature_list = []

for root, dirs, files in os.walk(images_dir):
    for file in tqdm(files):  # Iterate over the files in the directory
        img_path = os.path.join(root, file)
        feature = extract_feature(img_path, model)
        if feature is not None:
            filenames.append(img_path)
            feature_list.append(feature)

# Save the feature list and filenames to pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))


# Extract features
for img_path in tqdm(filenames):
            feature = extract_feature(img_path, model)
            if feature is not None:
                feature_list.append(feature)
                pickle.dump(feature_list,open('embeddings.pkl', 'wb' ))
                pickle.dump(filenames, open('filenames.pkl', 'wb'))