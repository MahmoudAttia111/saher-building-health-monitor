import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# تحميل الموديل من ملف Pickle
with open("bdt.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Saher - Building Health Monitor", layout="centered")

st.title("🏢 ساهر - نظام ذكي لمراقبة المباني")
st.markdown("هذا التطبيق يتنبأ بحالة المبنى (آمن / خطر / حرج) بناءً على بيانات المستشعرات.")

# إدخال البيانات من المستخدم
accel_x = st.number_input("📈 Accelerometer X (m/s²)", value=0.0)
accel_y = st.number_input("📉 Accelerometer Y (m/s²)", value=0.0)
accel_z = st.number_input("📊 Accelerometer Z (m/s²)", value=0.0)
strain = st.number_input("🧱 Strain (με)", value=0.0)
temp = st.number_input("🌡 Temperature (°C)", value=25.0)
seconds = st.number_input("⏱ Seconds (0–3600)", min_value=0, max_value=3600, value=0)

# معالجة البيانات
# total_accel = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
input_data = np.array([[  strain, temp, seconds,accel_x,accel_y,accel_z]])

# تطبيع البيانات
scaler = MinMaxScaler()
scaler.fit([[0,0,0,0], [50,5000,100,3600]])  # حدود تقريبية لكل عمود
input_scaled = scaler.transform(input_data)

# التنبؤ
if st.button("🔍 تنبؤ بحالة المبنى"):
    prediction = model.predict(input_scaled)

    # لو الموديل بيرجع أرقام مباشرة (تصنيف)
    if prediction.ndim == 1 or prediction.shape[1] == 1:
        label = int(prediction[0])
    else:
        label = int(np.argmax(prediction, axis=1)[0])

    # عرض النتيجة
    if label == 0:
        st.success("✅ المبنى آمن (Safe)")
    elif label == 1:
        st.warning("⚠️ المبنى في حالة خطر (Warning)")
    else:
        st.error("🚨 المبنى في حالة حرجة (Critical)")

    st.write("📊 Prediction Output:", prediction)
