# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:52:40 2023

@author: shrey
"""

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import sklearn
import io
from PIL import Image, ImageOps
import streamlit as st
from pathlib import Path


st.title("Fashion-MNIST Image Predictor")
file_path = Path(r"C:\Users\shrey\Downloads\Fashion-MNIST\fashion_mnist_cnn.h5")
model = load_model(file_path)
classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boots']


def main():
    select_page = st.sidebar.selectbox('Predict or About', ('Predict', 'About'))
    if (select_page == 'Predict'):
        predict()
        
    else:
        about()
            
def predict():

    uploaded_image = st.file_uploader("Drop an image here...", type = ['png', 'jpg'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        imageInput = image.resize((28, 28)) # condition for Fashion-MNIST
        imageInput = ImageOps.grayscale(imageInput)
        imageInput = ImageOps.invert(imageInput) # making the background black color
        # imageArray = np.array(imageInput)
        # normalized_image = imageArray / 255.0
        gray_input_image = np.reshape(imageInput, (-1, 28, 28, 1)) # condition for keras CNN model
        predicted_probs = model.predict(gray_input_image)
    
        index_max = np.argmax(predicted_probs, axis = -1)
        max_value = np.max(predicted_probs, axis = -1)
        max_value_percentage = str(max_value * 100) + "%"
        index_max = list(index_max)
        
        if (st.button("Predict")):
            st.success(f"Predicted class is {classes[index_max[0]]}")
            st.success(f"The model is {max_value_percentage} confident")
            if (st.button("Clear")):
                st.empty()
            
def about():
    st.title('About Page')
    st.write("Input any png, jpg, jpeg image from the web, and the model will predict one of the classes associated with the Fashion-MNIST dataset, along with it's probability.")
    st.write("However, do note that the model is far from being perfect; further changes regarding the model's accuracy will be made over time.")
    
    st.subheader("How this model works")
    st.write("Before you check the code for this model, you might be interested on how it recognizes images.")

    st.markdown(
"""
This is how the model was created:
- All the images in the training and testing data were converted to grayscale. Instead of three values (R, G, B), it is one value that represents the intensity of the pixel.
- The Pillow library was extremely helpful in converting downloaded images to grayscale. Image.invert() was important to change the invert the colors in the images to a black background.
"""
)
    
    
    st.subheader("Accuracy of the Model")
    st.markdown(
"""
Information about the accuracy of the model, and future improvements:
- Through the CNN model that the images were trained on had a 92 percent accuracy, it is actually quite difficult to predict a class for an image.
- For example, a T-Shirt, Shirt, and a Pullover might have the same pixel values when passed into the CNN model. 
- The model might fail with images that are in the RGBA, CMYK, or 1 mode. 
"""
)
    

        
    
            



if __name__  == '__main__':
    main()
        