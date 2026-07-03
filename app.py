from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("model/crop_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['N']),
        float(request.form['P']),
        float(request.form['K']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    prediction = model.predict([data])
    crop = encoder.inverse_transform(prediction)

    return render_template("result.html", prediction=crop[0])

if __name__ == "__main__":
    app.run(debug=True)
joblib.dump(model, "model/crop_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("Model Saved Successfully!")