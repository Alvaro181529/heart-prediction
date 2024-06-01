from flask import Flask, request, render_template
import pandas as pd
from model import predict, train_and_save_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
            ejection_fraction = float(request.form['ejection_fraction'])
            platelets = float(request.form['platelets'])
            serum_creatinine = float(request.form['serum_creatinine'])
            serum_sodium = float(request.form['serum_sodium'])

            data = pd.DataFrame([[age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium]],
                                columns=['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium'])

            result = predict(data)
            return render_template('result.html', prediction=result[0])
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == '__main__':
    # Entrenar y guardar el modelo al iniciar la aplicación
    train_and_save_model()
    app.run(debug=True)
