from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@app.route("/")
def home():
    return "✅ Flask API is running!"

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    messages = data.get("messages", [])
    results = []

    for msg in messages:
        result = sentiment_pipeline(msg[:512])[0]
        results.append({
            "message": msg,
            "sentiment": result["label"],
            "score": round(result["score"], 2)
        })

    return jsonify({"results": results})
