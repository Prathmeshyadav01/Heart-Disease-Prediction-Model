import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time

# Load the trained model
model_path = "rf_regressor.pkl"
scaler_x_path = "scaler_X.pkl"
scaler_y_path = "scaler_y.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Ensure it is in the correct directory.")
    st.stop()

model = joblib.load(model_path)
scaler_X = joblib.load(scaler_x_path) if os.path.exists(scaler_x_path) else None
scaler_y = joblib.load(scaler_y_path) if os.path.exists(scaler_y_path) else None

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Risk Prediction")

st.markdown("### Enter Patient Details")

# Creating layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 120, 30)
    sex = st.radio("Sex", ["Female (0)", "Male (1)"])
    sex = 0 if "Female" in sex else 1
    restingBP = st.slider("Resting BP", 50, 200, 120)
    chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])

with col2:
    thal = st.selectbox("Thal (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)", [0, 1, 2])
    major_vessels = st.slider("Major Vessels (0-3)", 0, 3, 1)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    cholesterol = st.slider("Cholesterol", 100, 600, 200)
    max_heart_rate = st.slider("Max Heart Rate", 60, 220, 150)

# Predict function
if st.button("🔍 Predict Risk"):
    with st.spinner("Analyzing... Please wait"):
        time.sleep(2)  # Simulating processing time
        features = np.array([
            age, sex, restingBP, thal, major_vessels, chest_pain_type, oldpeak, cholesterol, max_heart_rate
        ]).reshape(1, -1)

        # Scale features if necessary
        if scaler_X:
            features = scaler_X.transform(features)

        prediction = model.predict(features)

        # Inverse transform the prediction if necessary
        risk_score = prediction[0]
        st.write(risk_score)
        if scaler_y:
            risk_score = scaler_y.inverse_transform([[risk_score]])[0][0]

        # Show prediction result
        st.subheader("🔬 Prediction Result")
        risk_text = "High ⚠️" if prediction[0] == 1 else "Low ✅"
        st.write(f"**Risk Level:** {risk_text}")
        st.progress(float(risk_score))  # Visual progress bar for risk score
        st.write(f"**Risk Score:** {round(float(risk_score) * 100, 2)}%")

        # Additional recommendations
        st.markdown("### 🏥 Health Recommendations:")
        if prediction[0] == 1:
            st.warning("🚨 High risk detected! Consider consulting a doctor.")
            st.write("🔹 Maintain a balanced diet 🍏")
            st.write("🔹 Exercise regularly 🏃‍♂️")
            st.write("🔹 Monitor cholesterol levels ⚕️")
        else:
            st.success("✅ Your heart health seems good! Keep up with a healthy lifestyle. 🥦")
