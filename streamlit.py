import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import pandas as pd

with open('D:\College\Sem-1\FML\Week-15\Fruit_Image_Classification\\fruits_nutrition.json', 'r') as file:
    nutritional_values = json.load(file)

test_subset="D:\College\Sem-1\FML\Week-15\Fruit_Image_Classification\\filtered_dataset\\test"

BATCH_SIZE = 32
IMAGE_SIZE = 100

# test dataset pipeline
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
   test_subset,
    seed=42,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

#print training labels
ts_class_names = test_dataset.class_names


with open('D:\College\Sem-1\FML\Week-15\Fruit_Image_Classification\project_cnn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set the title of the app
st.title('Fruit Classification App')
st.text("~Nutritional Values and Fun Facts")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image and prepare it for classification
    img = load_img(uploaded_file, target_size=(100, 100))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  
    predicted_class_name = ts_class_names[predicted_class_index]

    st.write(f'Predicted Class: {ts_class_names[predicted_class[0]]}')

    if predicted_class_name in nutritional_values:
        st.write("Nutritional Value (per 100g):")
        nutritional_info = nutritional_values[predicted_class_name]
        nutritional_df = pd.DataFrame(nutritional_info.items(), columns=['Nutrient', 'Value'])
        st.table(nutritional_df)
    else:
        st.write("Nutritional information not available.")
