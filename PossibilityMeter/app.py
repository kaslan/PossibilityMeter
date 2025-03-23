
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Makine öğrenimi modelini yükle
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tahmin", methods=["POST"])
def tahmin_yap():
    data = request.json
    yeni_veri = pd.DataFrame([data])
    tahmin = model.predict(yeni_veri)
    return jsonify({"tahmin": int(tahmin[0])})

if __name__ == "__main__":
    app.run(debug=True)
