import streamlit as st
import numpy as np
import pickle
import joblib


with open("best_decision_tree.pkl", "rb") as f:
    model = joblib.load("best_decision_tree.pkl")


with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Saher - Building Health Monitor", layout="centered")

st.title("🏢 ساهر - نظام ذكي لمراقبة المباني")
st.markdown("هذا التطبيق يتنبأ بحالة المبنى (آمن / خطر / حرج) بناءً على بيانات المستشعرات.")

accel_x = st.number_input("📈 Accelerometer X (m/s²)", value=0.0)
accel_y = st.number_input("📉 Accelerometer Y (m/s²)", value=0.0)
accel_z = st.number_input("📊 Accelerometer Z (m/s²)", value=0.0)
strain = st.number_input("🧱 Strain (με)", value=0.0)
temp = st.number_input("🌡 Temperature (°C)", value=25.0)

input_data = np.array([[accel_x, accel_y, accel_z, strain, temp]])

input_scaled = scaler.transform(input_data)

if st.button("🔍 تنبؤ بحالة المبنى"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("✅ المبنى آمن (Safe)")
    elif prediction == 1:
        st.warning("⚠️ المبنى في حالة خطر (Warning)")
    else:
        st.error("🚨 المبنى في حالة حرجة (Critical)")

    st.write("🔢 التوقع (Label):", prediction)
