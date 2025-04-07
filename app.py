import streamlit as st
import pickle
import numpy as np

# Load the trained model and encoders
with open("C:\\Users\\rksri\\predict_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data['model']
    le_mainroad = data['le_mainroad']
    le_location = data['le_location']

st.title("🏠 House Price Predictor")

# Input form
with st.form("input_form"):
    area = st.number_input("Area (in sqft):", min_value=100)
    bedrooms = st.number_input("Bedrooms (BHK):", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Bathrooms:", min_value=1, max_value=10, step=1)
    stories = st.number_input("Stories:", min_value=1, max_value=10, step=1)
    mainroad = st.selectbox("Is the house on the main road?", ["yes", "no"])
    parking = st.number_input("Parking (no. of spaces):", min_value=0, max_value=5, step=1)
    location = st.selectbox("Location:", le_location.classes_)

    submit = st.form_submit_button("Predict Price")

if submit:
    # Encode inputs
    mainroad_encoded = le_mainroad.transform([mainroad])[0]
    location_encoded = le_location.transform([location])[0]

    input_array = np.array([[area, bedrooms, bathrooms, stories, mainroad_encoded, parking, location_encoded]])
    prediction = model.predict(input_array)[0]
    
    st.success(f"🏷️ Estimated House Price: ₹ {round(prediction, 2):,}")
