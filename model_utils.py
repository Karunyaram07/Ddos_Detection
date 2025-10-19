# model_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

# Same required columns from validation phase
REQUIRED_COLUMNS = ["Src_IP", "Dst_IP", "Protocol", "Packet_Size", "Duration"]

def preprocess_data(df, save_scaler: bool = False, model_dir: str = "model"):
    """
    Cleans, encodes, and normalizes dataset for training or detection.
    Returns: (X_scaled, scaler)
    """
    # Step 1: Drop NaN rows (cleaning)
    df = df.dropna(subset=["Protocol", "Packet_Size", "Duration"]).reset_index(drop=True)

    # Step 2: Encode categorical columns
    if "Protocol" in df.columns:
        encoder = LabelEncoder()
        df["Protocol"] = encoder.fit_transform(df["Protocol"])
    else:
        df["Protocol"] = 0  # fallback if missing (shouldn't happen after validation)

    # Step 3: Select numeric columns for model input
    numeric_cols = ["Protocol", "Packet_Size", "Duration"]

    # Convert to numeric (again, safe)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 4: Handle NaN after conversion (drop or fill)
    df = df.dropna(subset=numeric_cols)

    # Step 5: Normalize numeric data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # Optionally save scaler for reuse in detection
    if save_scaler:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    return X_scaled, scaler


def load_scaler(model_dir: str = "model"):
    """
    Loads saved scaler for detection phase.
    """
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please train the model first.")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)
