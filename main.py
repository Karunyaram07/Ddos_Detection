from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "secret123"  # Needed for flash messages

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        file = request.files["dataset"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            flash("Dataset uploaded successfully! Training would start here 🚀", "success")
            # Later: call autoencoder training function
            return redirect(url_for("results"))
    return render_template("train.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files["testdata"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            flash("Test dataset uploaded. Running anomaly detection ⚡", "info")
            # Later: call detection function and return results
            return redirect(url_for("results"))
    return render_template("detect.html")

@app.route("/results")
def results():
    # Placeholder — later show graphs/metrics
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)

