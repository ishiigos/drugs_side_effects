# app.py
import os
import joblib
import pandas as pd
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?export=download&id=158o3lNYzaETd7J9rAl7ogwtdHwmkY39_"
FEATURES_URL = "https://drive.google.com/uc?export=download&id=1iIy8DwC_bsN0Fd8JtYHqSBj2a-h3T-Yx"
LABELS_URL = "https://drive.google.com/uc?export=download&id=1hdAKyhUhrmiaFAUjMeIZc6k_znoch7tk"

# Local model path
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LABELS_PATH = "models/target_labels.pkl"

# Download model components if not already present
def download_if_needed(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

download_if_needed(MODEL_URL, MODEL_PATH)
download_if_needed(FEATURES_URL, FEATURES_PATH)
download_if_needed(LABELS_URL, LABELS_PATH)

# Load model and metadata
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
target_labels = joblib.load(LABELS_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = {}
    if request.method == "POST":
        user_input = {name: request.form.get(name, "") for name in feature_names}
        input_df = pd.DataFrame([user_input])
        try:
            preds = model.predict(input_df)
            prediction = [label for i, label in enumerate(target_labels) if preds[0][i] == 1]
        except Exception as e:
            prediction = [f"Error: {e}"]

    return render_template("index.html", prediction=prediction, features=feature_names, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)