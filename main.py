from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np

from data_utils import allowed_file_extension, validate_dataset, save_uploaded_file
from model_utils import preprocess_data
from model_train import train_autoencoder_from_npy
from model_detect import detect_anomalies

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"
RESULTS_FOLDER = os.path.join("static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("train"))

        if not allowed_file_extension(file.filename):
            flash("File type not allowed. Please upload .csv or .xlsx.", "danger")
            return redirect(url_for("train"))

        saved_path = save_uploaded_file(file, UPLOAD_FOLDER)

        valid, df_clean, message = validate_dataset(saved_path)
        if not valid:
            flash(f"Validation failed: {message}", "danger")
            return redirect(url_for("train"))

        try:
            X_scaled, scaler = preprocess_data(df_clean, save_scaler=True)
            npy_path = os.path.join(UPLOAD_FOLDER, "X_train_scaled.npy")
            np.save(npy_path, X_scaled)
            flash("✅ Dataset preprocessed successfully! Training started...", "info")

            train_info = train_autoencoder_from_npy(
                npy_path,
                model_dir=MODEL_FOLDER,
                results_dir=RESULTS_FOLDER,
                epochs=50,
                batch_size=32,
                encoding_dim=8,
                patience=6,
                verbose=1
            )
            flash(f"✅ Training complete! Model saved.", "success")
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"❌ Error during preprocessing or training: {e}", "danger")
            return redirect(url_for("train"))

    return render_template("train.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files.get("testdata")

        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("detect"))

        if not allowed_file_extension(file.filename):
            flash("File type not allowed. Please upload .csv or .xlsx.", "danger")
            return redirect(url_for("detect"))

        saved_path = save_uploaded_file(file, UPLOAD_FOLDER)

        valid, df_clean, message = validate_dataset(saved_path)
        if not valid:
            flash(f"Validation failed: {message}", "danger")
            return redirect(url_for("detect"))

        cleaned_path = os.path.join(UPLOAD_FOLDER, "test_cleaned.csv")
        df_clean.to_csv(cleaned_path, index=False)

        try:
            detect_info = detect_anomalies(
                test_path=cleaned_path,
                model_dir=MODEL_FOLDER,
                results_dir=RESULTS_FOLDER
            )

            flash(f"✅ Detection complete! {detect_info['anomalies']} anomalies found.", "success")
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"❌ Detection failed: {e}", "danger")
            return redirect(url_for("detect"))

    return render_template("detect.html")


@app.route("/results")
def results():
    summary_path = os.path.join(RESULTS_FOLDER, "detection_summary.txt")
    summary_text = open(summary_path).read() if os.path.exists(summary_path) else None

    error_hist_file = "results/error_hist.png" if os.path.exists(os.path.join("static", "results", "error_hist.png")) else None
    csv_file = "results/detection_results.csv" if os.path.exists(os.path.join("static", "results", "detection_results.csv")) else None

    return render_template("results.html", summary_text=summary_text, error_hist_file=error_hist_file, csv_file=csv_file)


if __name__ == "__main__":
    app.run(debug=True)
