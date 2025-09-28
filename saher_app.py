# saher_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


model = load_model("saher_normal_data.h5")

st.set_page_config(page_title="Saher - Building Health Monitor", layout="centered")

st.title("🏢 ساهر - نظام ذكي لمراقبة المباني")
st.markdown("هذا التطبيق يتنبأ بحالة المبنى (آمن / خطر / حرج) بناءً على بيانات المستشعرات.")


accel_x = st.number_input("📈 Accelerometer X (m/s²)", value=0.0)
accel_y = st.number_input("📉 Accelerometer Y (m/s²)", value=0.0)
accel_z = st.number_input("📊 Accelerometer Z (m/s²)", value=0.0)
strain = st.number_input("🧱 Strain (με)", value=0.0)
temp = st.number_input("🌡 Temperature (°C)", value=25.0)


total_accel = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
input_data = np.array([[total_accel, strain, temp]])

scaler = MinMaxScaler()

scaler.fit([[0,0,0],[50,5000,100]])
input_scaled = scaler.transform(input_data)


if st.button("🔍 تنبؤ بحالة المبنى"):
    prediction = model.predict(input_scaled)
    label = np.argmax(prediction)

    if label == 0:
        st.success("✅ المبنى آمن (Safe)")
    elif label == 1:
        st.warning("⚠️ المبنى في حالة خطر (Warning)")
    else:
        st.error("🚨 المبنى في حالة حرجة (Critical)")

    st.write("📊 Probabilities:", prediction)
