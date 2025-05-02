import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Load preprocessor dan model TFLite
preprocessor = joblib.load("preprocessor.pkl")
interpreter = tf.lite.Interpreter(model_path="laptop_price_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Klasifikasi Harga Laptop")
st.write("Masukkan spesifikasi laptop untuk memprediksi kelas harga: Murah, Sedang, atau Mahal.")

# Input form
brand = st.selectbox("Brand", ["Acer", "Asus", "Dell", "HP", "Lenovo", "MSI", "Apple", "Samsung"])  # Sesuaikan jika diperlukan
processor_speed = st.number_input("Kecepatan Prosesor (GHz)", min_value=0.5, max_value=5.0, value=2.5)
ram_size = st.number_input("Ukuran RAM (GB)", min_value=2, max_value=64, value=8)
storage_capacity = st.number_input("Kapasitas Penyimpanan (GB)", min_value=128, max_value=2048, value=512)
screen_size = st.number_input("Ukuran Layar (Inch)", min_value=10.0, max_value=20.0, value=15.6)
weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=1.8)

if st.button("Prediksi Kategori Harga"):
    # Buat dataframe input
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Processor_Speed": processor_speed,
        "RAM_Size": ram_size,
        "Storage_Capacity": storage_capacity,
        "Screen_Size": screen_size,
        "Weight": weight
    }])

    # Transformasi fitur
    input_transformed = preprocessor.transform(input_df).astype(np.float32)

    # Prediksi dengan model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_transformed)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Mapping kelas ke label
    class_mapping = {0: "Murah", 1: "Sedang", 2: "Mahal"}
    st.success(f"Prediksi: **{class_mapping[predicted_class]}**")
