import streamlit as st
import requests

st.title('Wine Classifier')

# Create input fields
features = [st.text_input(f"Feature {i}", "0") for i in range(13)]

if st.button('Predict'):
    # Make a request to FastAPI
    response = requests.post("http://api:8000/predict/", json={"features": [float(i) for i in features]})
    prediction = response.json().get('prediction')
    st.write(f"The predicted class is: {prediction}")
