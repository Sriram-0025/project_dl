
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
import streamlit as st

# Define options
options = [0, 1, 2, 3]

# Multiple selection for each feature
thinning = st.multiselect('Thinning of the suprapapillary epidermis', options, default=[0])
clubbing = st.multiselect('Clubbing of the rete ridges', options, default=[0])
spongiosis = st.multiselect('Spongiosis', options, default=[0])
fibrosis = st.multiselect('Fibrosis of the papillary dermis', options, default=[0])
koebner = st.multiselect('Koebner phenomenon', options, default=[0])
elongation = st.multiselect('Elongation of the rete ridges', options, default=[0])
exocytosis = st.multiselect('Exocytosis', options, default=[0])
melanin = st.multiselect('Melanin incontinence', options, default=[0])
pnl_infiltrate = st.multiselect('Pnl infiltrate', options, default=[0])
saw_tooth = st.multiselect('Saw-tooth appearance of retes', options, default=[0])


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
