from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Extract user input
        age = request.form.get("age", type=float)
        gender = request.form.get("gender")
        bmi = request.form.get("bmi", type=float)

        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "bmi": bmi
        }])

        # Preprocess and predict
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)