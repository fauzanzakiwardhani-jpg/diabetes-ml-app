import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Masukkan data pasien untuk memprediksi risiko diabetes.")

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

df = load_data()

# =====================
# PREPROCESSING
# =====================
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

df["gender"] = gender_encoder.fit_transform(df["gender"])
df["smoking_history"] = smoking_encoder.fit_transform(df["smoking_history"])


X = df.drop("diabetes", axis=1)
y = df["diabetes"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# TRAIN MODEL (ONCE)
# =====================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_scaled, y)

# =====================
# INPUT FORM (UI)
# =====================
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 1, 120, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking = st.selectbox(
    "Smoking History",
    smoking_encoder.classes_
    )
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 50, 300, 120)

    submitted = st.form_submit_button("🔍 Predict")

# =====================
# PREDICTION
# =====================
if submitted:
    gender_enc = gender_encoder.transform([gender])[0]
    smoking_enc = smoking_encoder.transform([smoking])[0]

    input_data = np.array([[
        gender_enc,
        age,
        hypertension,
        heart_disease,
        smoking_enc,
        bmi,
        hba1c,
        glucose
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Berpotensi Diabetes (Probabilitas: {probability:.2%})")
    else:
        st.success(f"✅ Tidak Diabetes (Probabilitas: {1 - probability:.2%})")
