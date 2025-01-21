import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image


model = load_model('Age_Sex_detection.h5')

def preprocess_image(image):
    image = image.convert('RGB')  
    image = image.resize((48, 48))  
    img_array = np.array(image)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    return img_array

def predict_age_gender(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    gender_labels = ['Male', 'Female']  
    age = int(np.round(predictions[1][0]))  
    sex = int(np.round(predictions[0][0]))  
    gender = gender_labels[sex]  
    return age, gender

st.title('Age and Gender Detection')
st.subheader('Upload an image for age and gender prediction')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=False,width=250)
    age, gender = predict_age_gender(image)

    
    st.markdown(f"<h2 style='color:blue;'>Predicted Age: {age}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:green;'>Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
