import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tempfile

# Load the trained model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("aquatic_species_model.keras")
    return model

@st.cache_resource
def load_label_encoder():
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)
    return label_encoder

model = load_model()
label_encoder = load_label_encoder()
class_labels = label_encoder.classes_

# Prediction function
def predict_species(audio_bytes):
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load audio
        audio, sample_rate = librosa.load(tmp_path, sr=16000)

        # Pad if too short
        min_length = 2048
        if len(audio) < min_length:
            pad_length = min_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')

        # Extract MFCC
        n_fft = min(2048, len(audio) // 2)
        hop_length = n_fft // 4
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60, n_fft=n_fft, hop_length=hop_length)
        mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1, 1)

        # Predict
        prediction = model.predict(mfccs_scaled)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_label = class_labels[predicted_class]

        return f"✅ Predicted Species: **{predicted_label}**\n🎯 Confidence: **{confidence:.2f}%**"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Aquatic Species Classifier", page_icon="🌊")
st.title("🌊 Aquatic Species Audio Classifier")
st.write("Upload a WAV file of an aquatic sound to predict its species using a deep learning model.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("🔍 Classify"):
        with st.spinner("Analyzing audio..."):
            result = predict_species(uploaded_file.read())
        st.markdown(result)
else:
    st.info("Please upload a `.wav` file to get started.")
