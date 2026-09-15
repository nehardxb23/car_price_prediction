from flask import Flask, request, render_template_string
import json
from urllib.request import Request, urlopen

app = Flask(__name__)

MODEL_URL = "http://127.0.0.1:5001/invocations"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Price Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            margin: 0;
            padding: 40px 20px;
        }

        .container {
            background: white;
            padding: 35px;
            border-radius: 20px;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        h1 {
            text-align: center;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 12px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 14px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #5568d8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #f1f5ff;
            border-radius: 10px;
            text-align: center;
        }

        .price {
            font-size: 30px;
            font-weight: bold;
            color: #667eea;
            margin-top: 8px;
        }

        .error {
            color: red;
            margin-top: 15px;
            text-align: center;
        }

        @media (max-width: 600px) {
            .row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <h1>🚗 Car Price Prediction</h1>

    <div class="subtitle">
        MLflow Model Deployment
    </div>

    <form method="POST">

        <div class="row">

            <div>
                <label>Brand</label>
                <input type="text" name="brand"
                       placeholder="e.g. Maruti" required>
            </div>

            <div>
                <label>Model</label>
                <input type="text" name="model"
                       placeholder="e.g. Swift" required>
            </div>

        </div>

        <div class="row">

            <div>
                <label>Vehicle Age (years)</label>
                <input type="number" name="vehicle_age"
                       min="0" required>
            </div>

            <div>
                <label>KM Driven</label>
                <input type="number" name="km_driven"
                       min="0" required>
            </div>

        </div>

        <div class="row">

            <div>
                <label>Seller Type</label>
                <select name="seller_type">
                    <option>Individual</option>
                    <option>Dealer</option>
                    <option>Trustmark Dealer</option>
                </select>
            </div>

            <div>
                <label>Fuel Type</label>
                <select name="fuel_type">
                    <option>Petrol</option>
                    <option>Diesel</option>
                    <option>CNG</option>
                    <option>LPG</option>
                    <option>Electric</option>
                </select>
            </div>

        </div>

        <div class="row">

            <div>
                <label>Transmission</label>
                <select name="transmission_type">
                    <option>Manual</option>
                    <option>Automatic</option>
                </select>
            </div>

            <div>
                <label>Mileage (km/l)</label>
                <input type="number" name="mileage"
                       step="0.1" min="0" required>
            </div>

        </div>

        <div class="row">

            <div>
                <label>Engine (cc)</label>
                <input type="number" name="engine"
                       min="0" required>
            </div>

            <div>
                <label>Max Power (bhp)</label>
                <input type="number" name="max_power"
                       step="0.1" min="0" required>
            </div>

        </div>

        <label>Seats</label>
        <input type="number" name="seats"
               min="2" max="10" required>

        <button type="submit">
            Predict Car Price
        </button>

    </form>

    {% if prediction is not none %}
    <div class="result">
        <div>Predicted Car Price</div>
        <div class="price">
            ₹{{ "{:,.0f}".format(prediction) }}
        </div>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
        {{ error }}
    </div>
    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            brand = request.form["brand"]
            model = request.form["model"]
            vehicle_age = float(request.form["vehicle_age"])
            km_driven = float(request.form["km_driven"])
            seller_type = request.form["seller_type"]
            fuel_type = request.form["fuel_type"]
            transmission_type = request.form["transmission_type"]
            mileage = float(request.form["mileage"])
            engine = float(request.form["engine"])
            max_power = float(request.form["max_power"])
            seats = float(request.form["seats"])

            payload = {
                "dataframe_split": {
                    "columns": [
                        "brand",
                        "model",
                        "vehicle_age",
                        "km_driven",
                        "seller_type",
                        "fuel_type",
                        "transmission_type",
                        "mileage",
                        "engine",
                        "max_power",
                        "seats"
                    ],

                    "data": [[
                        brand,
                        model,
                        vehicle_age,
                        km_driven,
                        seller_type,
                        fuel_type,
                        transmission_type,
                        mileage,
                        engine,
                        max_power,
                        seats
                    ]]
                }
            }

            request_data = Request(
                MODEL_URL,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urlopen(request_data) as response:

                result = json.loads(
                    response.read().decode()
                )

            prediction = result["predictions"][0]

        except Exception as e:

            print("ERROR:", e)

            error = "Could not connect to the MLflow model server."


    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(port=8000, debug=True)