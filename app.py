import os
import joblib
import pandas as pd
import requests
import re
from flask import Flask, render_template, request

app = Flask(__name__)

# Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?export=download&id=158o3lNYzaETd7J9rAl7ogwtdHwmkY39_"
FEATURES_URL = "https://drive.google.com/uc?export=download&id=1iIy8DwC_bsN0Fd8JtYHqSBj2a-h3T-Yx"
LABELS_URL = "https://drive.google.com/uc?export=download&id=1hdAKyhUhrmiaFAUjMeIZc6k_znoch7tk"

# Local paths
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LABELS_PATH = "models/target_labels.pkl"

# --- Helper: Safe Google Drive downloader ---
def get_confirm_token(response_text):
    match = re.search(r'confirm=([0-9A-Za-z_]+)', response_text)
    return match.group(1) if match else None

def download_from_google_drive(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Found existing: {dest_path}")
        return

    print(f"Downloading {dest_path} from Google Drive...")
    session = requests.Session()
    response = session.get(url, stream=True)
    token = get_confirm_token(response.text)

    if token:
        url += f"&confirm={token}"
        response = session.get(url, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Download models if needed ---
download_from_google_drive(MODEL_URL, MODEL_PATH)
download_from_google_drive(FEATURES_URL, FEATURES_PATH)
download_from_google_drive(LABELS_URL, LABELS_PATH)

# --- Load assets ---
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
            prediction = [f"Error: {str(e)}"]

    return render_template("index.html", prediction=prediction, features=feature_names, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)