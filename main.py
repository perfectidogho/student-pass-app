from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("student_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[f]) for f in ['age', 'studytime', 'failures', 'absences']]
        df = pd.DataFrame([input_data], columns=['age', 'studytime', 'failures', 'absences'])
        prediction = model.predict(df)[0]
        result = "PASS" if prediction == 1 else "FAIL"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
  
