import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Auto MPG Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    .prediction-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🚗 Auto MPG Prediction")
st.markdown("""
    Predict the fuel efficiency (Miles Per Gallon) of a vehicle based on its technical specifications.
    This model was trained on the classic Auto MPG dataset.
""")

# Load Model
MODEL_PATH = 'model/auto_mpg_prediction_model.joblib'
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

model = load_model()

# Median horsepower from training data (used for missing values)
MEDIAN_HP = 93.5

# Input Form
with st.container():
    st.subheader("Vehicle Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.number_input("Displacement (cu. in.)", min_value=60.0, max_value=500.0, value=150.0, step=0.1)
        horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=MEDIAN_HP, step=1.0, help="Leave as 93.5 if unknown")
    
    with col2:
        weight = st.number_input("Weight (lbs)", min_value=1500, max_value=6000, value=3000, step=1)
        acceleration = st.number_input("Acceleration (0-60 mph sec)", min_value=8.0, max_value=25.0, value=15.0, step=0.1)
        model_year = st.number_input("Model Year (e.g., 70 for 1970)", min_value=70, max_value=82, value=76, step=1)

    origin = st.radio("Origin", ["USA", "Europe", "Asia"], horizontal=True)

# Prediction Logic
if st.button("Predict MPG"):
    if model:
        # 1. Feature Engineering
        power_to_weight = horsepower / weight
        
        # 2. One-hot encoding for Origin
        origin_USA = 1 if origin == "USA" else 0
        origin_Europe = 1 if origin == "Europe" else 0
        origin_Asia = 1 if origin == "Asia" else 0
        
        # 3. Create input DataFrame (matching the structure used in training)
        # Order: ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin_USA', 'origin_Europe', 'origin_Asia', 'power_to_weight']
        input_data = pd.DataFrame([[
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            model_year,
            origin_USA,
            origin_Europe,
            origin_Asia,
            power_to_weight
        ]], columns=[
            'cylinders', 'displacement', 'horsepower', 'weight', 
            'acceleration', 'model year', 'origin_USA', 'origin_Europe', 'origin_Asia', 'power_to_weight'
        ])
        
        # 4. Make Prediction
        prediction = model.predict(input_data)[0]
        
        # 5. Display Result
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Fuel Efficiency</h3>
                <div class="prediction-value">{prediction:.2f} MPG</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.balloons()
    else:
        st.warning("Please ensure the model is loaded correctly.")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit & Scikit-Learn")
