import joblib
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io

# Load trained model and label encoder
model = joblib.load('music_genre_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Sample audio files for testing
sample_files = {
    "Sample1": "samples/rock.mp3",
    "Sample2": "samples/jazz.mp3",
    "Sample3": "samples/hiphop.mp3",
    "Sample4": "samples/classical.mp3",
    "Sample5": "samples/pop.mp3",
    "Sample6": "samples/blues.mp3",
    "Sample7": "samples/country.mp3",
    "Sample8": "samples/electronic.mp3",
    "Sample9": "samples/reggae.mp3",
    "Sample10": "samples/metal.mp3"
}

# Function to extract features from audio file
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.markdown("Upload an audio file or select a sample to classify its genre.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

# Sample audio selection
selected_sample = st.selectbox("Or try a sample file:", ["None"] + list(sample_files.keys()))

# Load and play selected sample
if selected_sample != "None":
    sample_path = sample_files[selected_sample]
    audio_data, sr = librosa.load(sample_path, sr=None)
    st.audio(sample_path, format='audio/wav')
    features = extract_features(audio_data, sr)
    classify = True

elif uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    st.audio(uploaded_file, format='audio/wav')
    features = extract_features(audio_data, sr)
    classify = True
else:
    classify = False

if classify:
    # Predict genre
    prediction = model.predict(features)
    predicted_genre = label_encoder.inverse_transform(prediction)[0]
    
    # Display prediction
    st.success(f"🎶 Predicted Genre: **{predicted_genre}**")
    
    # Display waveform
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax, alpha=0.7)
    ax.set_title("Waveform of the Audio")
    st.pyplot(fig)
    
    # Display spectrogram
    fig, ax = plt.subplots(figsize=(8, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title("Spectrogram")
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)
