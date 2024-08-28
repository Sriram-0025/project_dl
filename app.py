
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the scaler
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)

# Load the model
model_path = 'model.keras'
model = load_model(model_path)

# Define class labels
class_labels = {
    1: 'psoriasis',
    2: 'seboreic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'chronic dermatitis',
    6: 'pityriasis rubra pilaris'
}

# Streamlit App
st.title('Dermatology Classification')

st.write("Enter the feature values for prediction:")

# Inputs
thinning = st.number_input('Thinning of the suprapapillary epidermis', min_value=0, max_value=3)
clubbing = st.number_input('Clubbing of the rete ridges', min_value=0, max_value=3)
spongiosis = st.number_input('Spongiosis', min_value=0, max_value=3)
fibrosis = st.number_input('Fibrosis of the papillary dermis', min_value=0, max_value=3)
koebner = st.number_input('Koebner phenomenon', min_value=0, max_value=3)
elongation = st.number_input('Elongation of the rete ridges', min_value=0, max_value=3)
exocytosis = st.number_input('Exocytosis', min_value=0, max_value=3)
melanin = st.number_input('Melanin incontinence', min_value=0, max_value=3)
pnl_infiltrate = st.number_input('Pnl infiltrate', min_value=0, max_value=3)
saw_tooth = st.number_input('Saw-tooth appearance of retes', min_value=0, max_value=3)

# Collect input data
new_input_data = np.array([[thinning, clubbing, spongiosis, fibrosis, koebner,
                            elongation, exocytosis, melanin, pnl_infiltrate, saw_tooth]])

# Scale the data
new_input_data_scaled = scaler.transform(new_input_data)

# Make prediction
if st.button('Predict'):
    predictions = model.predict(new_input_data_scaled)
    predicted_class_index = np.argmax(predictions, axis=-1) + 1
    predicted_class_label = class_labels.get(predicted_class_index[0], 'Unknown')
    
    st.write(f"Predicted class index: {predicted_class_index[0]}")
    st.write(f"Predicted class label: {predicted_class_label}")
