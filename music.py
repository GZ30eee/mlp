import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = joblib.load('music_genre_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract features from the uploaded audio file
def extract_features(audio_file):
    # Placeholder for feature extraction logic (use librosa or similar)
    # Return a numpy array of features (for now using random features as an example)
    return np.random.rand(1, 13)  # Replace with actual feature extraction logic

# Streamlit UI to upload file
st.title('Music Genre Classifier')
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Extract features from the uploaded audio file
    features = extract_features(uploaded_file)
    
    # Predict genre using the trained model
    prediction = model.predict(features)  # Predict genre (output will be numeric)

    # Convert numeric prediction back to genre using LabelEncoder
    predicted_genre = label_encoder.inverse_transform(prediction)

    # Show the prediction
    st.write(f'Predicted Genre: {predicted_genre[0]}')
