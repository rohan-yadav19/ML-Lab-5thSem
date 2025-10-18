import streamlit as st
import joblib

# Load model
model = joblib.load('mymodel.pkl')

# UI
st.title("📘 Predict Student Marks from Study Hours")
st.write("This app uses a simple Linear Regression model to predict marks based on hours studied.")

# Input
hours = st.number_input("Enter number of hours studied:", min_value=0.0, max_value=24.0, value=2.0, step=0.5)

# Predict
if st.button("Predict Marks"):
    prediction = model.predict([[hours]])[0]
    st.success(f"📊 Predicted Marks: {prediction:.2f}")
