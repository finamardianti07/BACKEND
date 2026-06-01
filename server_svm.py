from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from flask_cors import CORS
CORS(app)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

app = Flask(__name__)

# load model
model = joblib.load("model_svm.pkl")

categories = ["bercak", "Hawar", "Healthy"]

def preprocess_image(file):
    file_bytes = file.read()

    if not file_bytes:
        raise ValueError("File kosong")

    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Gambar tidak valid")

    img = cv2.resize(img, (64, 64))
    img = img.flatten()

    return img.reshape(1, -1)

@app.route("/")
def home():
    return "API SVM jalan 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        features = preprocess_image(file)

        pred = model.predict(features)
        proba = model.predict_proba(features)

        label = categories[pred[0]]
        confidence = float(proba.max() * 100)

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)