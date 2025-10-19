import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

REQUIRED_COLUMNS = ["Protocol", "Packet_Size", "Duration"]

def preprocess_data(df, save_scaler: bool = False, model_dir: str = "model"):
    """
    Encodes, fills missing values, and scales dataset.
    Returns: X_scaled, scaler
    """
    # Step 1: Fill missing numeric
    for col in ["Packet_Size", "Duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Step 2: Encode Protocol if present
    if "Protocol" in df.columns:
        encoder = LabelEncoder()
        df["Protocol"] = encoder.fit_transform(df["Protocol"])
    else:
        df["Protocol"] = 0  # fallback

    # Step 3: Select numeric columns available
    numeric_cols = [col for col in ["Protocol", "Packet_Size", "Duration"] if col in df.columns]
    X = df[numeric_cols].values

    # Step 4: Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if save_scaler:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    return X_scaled, scaler

def load_scaler(model_dir: str = "model"):
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please train the model first.")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)
