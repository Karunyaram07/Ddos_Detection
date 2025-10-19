import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def detect_anomalies(test_path, model_dir="model", results_dir="static/results", threshold_factor=2.0):
    """
    Detect anomalies using the trained Autoencoder model.
    Saves CSV results, histogram plot, and summary text.
    """
    # --- Ensure model and scaler exist ---
    model_path = os.path.join(model_dir, "autoencoder.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained model or scaler not found. Please train first.")

    model = load_model(model_path,compile=False)
    scaler = joblib.load(scaler_path)

    # --- Load and preprocess test dataset ---
    df = pd.read_csv(test_path)
    X_test = df.select_dtypes(include=[np.number]).values
    X_scaled = scaler.transform(X_test)

    # --- Predict reconstruction ---
    reconstructed = model.predict(X_scaled)
    reconstruction_error = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    # --- Calculate anomaly threshold ---
    threshold = np.mean(reconstruction_error) + threshold_factor * np.std(reconstruction_error)

    # --- Label anomalies ---
    df["Reconstruction_Error"] = reconstruction_error
    df["Anomaly"] = df["Reconstruction_Error"] > threshold

    # --- Save detection results CSV ---
    result_path = os.path.join(results_dir, "detection_results.csv")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(result_path, index=False)

    # --- Plot histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(reconstruction_error, bins=50, color="skyblue", alpha=0.7)
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.legend()
    plot_path = os.path.join(results_dir, "error_hist.png")
    plt.savefig(plot_path)
    plt.close()

    # --- Write summary file ---
    anomaly_count = int(df["Anomaly"].sum())
    total = len(df)
    summary_path = os.path.join(results_dir, "detection_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Threshold Used: {threshold:.4f}\\n")
        f.write(f"Total Records: {total}\\n")
        f.write(f"Anomalies Detected: {anomaly_count}\\n")

    # --- Return structured info ---
    return {
        "threshold": threshold,
        "anomalies": anomaly_count,
        "total": total,
        "result_csv": result_path,
        "plot_path": plot_path,
        "summary_file": summary_path
    }
