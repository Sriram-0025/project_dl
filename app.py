
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

import streamlit as st


# Dropdown menu for each feature
thinning = st.selectbox('Thinning of the suprapapillary epidermis', options)
clubbing = st.selectbox('Clubbing of the rete ridges', options)
spongiosis = st.selectbox('Spongiosis', options)
fibrosis = st.selectbox('Fibrosis of the papillary dermis', options)
koebner = st.selectbox('Koebner phenomenon', options)
elongation = st.selectbox('Elongation of the rete ridges', options)
exocytosis = st.selectbox('Exocytosis', options)
melanin = st.selectbox('Melanin incontinence', options)
pnl_infiltrate = st.selectbox('Pnl infiltrate', options)
saw_tooth = st.selectbox('Saw-tooth appearance of retes', options)



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
