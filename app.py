from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# load model and features
model = joblib.load("simple_student_model.pkl")  # This loads the model
features = joblib.load("features.pkl")           # This loads the feature list


@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(f)) for f in features]
        df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(df)[0]
        result = '🎓 PASS' if prediction == 1 else '❌ FAIL'
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)