import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load your trained model
model = tf.keras.models.load_model('Stuntify/my_model.h5')

# Load the scaler
scaler = joblib.load('Stuntify/scaler.pkl')

def main():
    # Add a custom style
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f9;
            color: #333333;
        }
        .main-title {
            font-family: 'Arial', sans-serif;
            font-size: 2.5rem;
            color: #cc68e3;
        }
        .subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 1.5rem;
            color: #cc68e3;
        }
        button {
            background-color: #1abc9c;
            color: white;
        }
        .sidebar-logo {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add a logo to the sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-logo'>", unsafe_allow_html=True)
        st.image("Stuntify/logo.png", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>Stuntify</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Masukkan Data untuk Melakukan Pengecekan Stunting</p>", unsafe_allow_html=True)

    # Input fields
    age = st.number_input("Umur (Bulan):", min_value=0, step=1, format="%d")
    birth_weight = st.number_input("Berat Lahir (KG):", min_value=0.0, step=0.1, format="%.1f")
    birth_length = st.number_input("Panjang Lahir (CM):", min_value=0.0, step=0.1, format="%.1f")
    body_weight = st.number_input("Berat Badan (KG):", min_value=0.0, step=0.1, format="%.1f")
    body_length = st.number_input("Tinggi Badan (CM):", min_value=0.0, step=0.1, format="%.1f")
    gender = st.radio("Jenis Kelamin:", ("L", "P"))

    if st.button("Lakukan Prediksi"):
        try:
            # Encode gender
            gender_encoded = 1 if gender == "L" else 0

            # Prepare input data
            input_data = np.array([[age, birth_weight, birth_length, body_weight, body_length, gender_encoded]])

            # Rescale input data
            rescaled_input = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(rescaled_input)

            # Interpret the result
            result = "Berisiko Stunting" if prediction[0][0] > 0.5 else "Tidak Berisiko Stunting"
            st.success(f"Hasil: {result}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
