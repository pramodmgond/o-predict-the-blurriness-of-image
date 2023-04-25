#custom web apps for machine learning and data science
import streamlit as st
import cv2
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np

models = keras.models.load_model("model1.h5") # load model

#page configuration of the Streamlit app
#specifies the title of the web page
#specifies the icon of the page
st.set_page_config(
    page_title="Image Classifier",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded"
)

target_size = (64, 64)
# allows the user to upload an image file
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

def measure_sharpness(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply the Laplacian operator to the grayscale image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # compute the variance of the Laplacian
    variance = laplacian.var()
    
    return variance

if uploaded_file is not None:

    # opened using the PIL library's Image.open function
    # resized to the target size
    # converted to a numpy array
    # batch dimension is added most machine learning models expect input data to have a batch dimension

    image = Image.open(uploaded_file)
    image = image.resize(target_size)
    image_array = np.array(image)
    
    image_array = np.expand_dims(image_array, axis=0)
    image_array =  image_array/255 
    
    sharpness = measure_sharpness(np.array(image))
    st.write("Sharpness of the Images is: {:.4f}".format(sharpness))

    # which returns an array of probabilities for each class
    # class with the highest probability
    # predicted class is displayed to the user
    y_predict = np.argmax(models.predict(image_array))
    if y_predict == 0:
        st.write('Artificially-Blurred Image')
    elif y_predict == 1:
        st.write('Naturally-Blurred Image')
    elif y_predict == 2:
        st.write('Undistorted Image')

    st.image(image, caption=f'Uploaded Image prediction ({y_predict})', width=200)
