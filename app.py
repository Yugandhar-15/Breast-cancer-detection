
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("🎗 Breast Cancer Detection")

radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")

features = np.array([[
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0
]])

if st.button("Predict"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ Benign Tumor (Non-Cancerous)")
    else:
        st.error("⚠ Malignant Tumor (Cancerous)")