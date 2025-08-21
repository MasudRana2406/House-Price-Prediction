from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------
# Load the model
# -------------------------
model = None
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'house_price_model_multi.pkl')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# -------------------------
# Flask routes
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_message = None

    if request.method == 'POST':
        if model is None:
            error_message = "Error: Prediction model not available."
        else:
            try:
                size = float(request.form.get('size', 0))
                bedrooms = int(request.form.get('bedrooms', 0))
                age = float(request.form.get('age', 0))

                features = np.array([[size, bedrooms, age]])
                pred_value = model.predict(features)[0]
                prediction = f"Predicted Price: ${pred_value:,.2f}"
            except ValueError:
                error_message = "Please enter valid inputs."

    return render_template('index01.html', prediction=prediction, error_message=error_message)

# -------------------------
# Run Flask app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
