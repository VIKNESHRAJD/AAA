import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder
import os

# Load trained model
model = tf.keras.models.load_model("aquatic_species_model.h5")

# Define species labels
labels = [
    "AtlanticSpottedDolphin", "CommonDolphin", "SpinnerDolphin", "Commerson'sDolphin", "StripedDolphin",
    "Dall'sPorpoise", "WeddellSeal", "Fin_FinbackWhale", "KillerWhale", "SpottedSeal", "NorthernRightWhale",
    "DuskyDolphin", "BeardedSeal", "HarborPorpoise", "BottlenoseDolphin", "GraySeal", "Short_Finned(Pacific)PilotWhale",
    "SpermWhale", "Narwhal", "Beluga_WhiteWhale", "MinkeWhale", "Fraser'sDolphin", "HarpSeal", "PantropicalSpottedDolphin",
    "RingedSeal", "TucuxiDolphin", "White_sidedDolphin", "White_beakedDolphin", "WestIndianManatee", "Walrus",
    "RossSeal", "SouthernRightWhale", "Boutu_AmazonRiverDolphin", "LeopardSeal", "JuanFernandezFurSeal",
    "SeaOtter", "HarbourSeal", "Long_FinnedPilotWhale", "LongBeaked(Pacific)CommonDolphin", "Grampus_Risso'sDolphin",
    "NewZealandFurSeal", "StellerSeaLion", "BlueWhale", "ClymeneDolphin", "HumpbackWhale", "Rough_ToothedDolphin",
    "GrayWhale", "HoodedSeal", "BowheadWhale", "MelonHeadedWhale", "RibbonSeal", "Heaviside'sDolphin",
    "FinlessPorpoise", "FalseKillerWhale", "IrawaddyDolphin"
]
le = LabelEncoder().fit(labels)

st.title("Aquatic Species Classifier")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    features = np.expand_dims(features, axis=(0, 2))
    prediction = model.predict(features)
    predicted_label = labels[np.argmax(prediction)]
    st.subheader(f"Predicted Species: {predicted_label}")
