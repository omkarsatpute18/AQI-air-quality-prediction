from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")

model = joblib.load(MODEL_PATH)

# Exact feature order used during model training
FEATURES = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Check that all required features are present
        for feature in FEATURES:
            if feature not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing input: {feature}"
                }), 400

        # Convert input values to float
        input_data = {}

        for feature in FEATURES:
            value = float(data[feature])

            if value < 0:
                return jsonify({
                    "success": False,
                    "error": f"{feature} cannot be negative."
                }), 400

            input_data[feature] = value

        # Create DataFrame in exact training feature order
        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        # Make prediction
        prediction = model.predict(input_df)

        aqi = float(prediction[0])

        # AQI should not be negative
        aqi = max(0, aqi)

        # Round for display
        aqi = round(aqi, 2)

        category = get_aqi_category(aqi)

        return jsonify({
            "success": True,
            "aqi": aqi,
            "category": category
        })

    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please enter valid numeric values."
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )