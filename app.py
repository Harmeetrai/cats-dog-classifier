import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import load_model
import numpy as np
import shutil

import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module

#=================================== Title ===============================
st.title("""
Cat 🐱 Or Dog 🐶 Recognizer
	""")

#================================= Title Image ===========================
st.text("""""")
img_path_list = ["static/img/cat_dog.jpg"]
image = Image.open(img_path_list[0])
st.image(
	        image,
	        use_column_width=True,
	    )

#================================= About =================================
st.write("""
##  About
	""")
st.write("""
Welcome to this project. It is a Cat Or Dog Recognizer App!
	""")
st.write("""
You have to upload your own test images to test it!
	""")

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview 👀 Of Given Image!
		""")
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		**Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! **
		""")
except:
	st.write("""
		### N0 Picture hasn't selected yet!
		""")

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#==================================== Model ==================================
def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path, 
                   target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array / 255.  # Preprocess the image
    prediction = loaded_model.predict(preprocessed_img)
    return prediction

def generate_result(prediction):
    
	st.write("""
	## 🎯 RESULT
		""")
	if prediction[0][0]>0.5:
	    st.write("""
	    	## Model predicts it as an image of a CAT 🐱!!!
	    	""")
	else:
	    st.write("""
	    	## Model predicts it as an image of a DOG 🐶!!!
	    	""")

#=========================== Predict Button Clicked ==========================
if submit:

    # save image on that directory
    save_img("static/uploads/test_image.png", img_array)

    image_path = "static/uploads/test_image.png"
    # Predicting
    st.write("👁️ Predicting...")

    loaded_model = load_model("static/models/model.h5")

    prediction = processing(image_path)

    cat_value = prediction[0][0] * 100
    dog_value = prediction[0][1] * 100

    generate_result(prediction)

    st.write("Chance of Cat: ", cat_value, " %")
    st.write("Chance of Dog: ", dog_value, " %")
