import pandas as pd
from flask import Flask,request,render_template,app,url_for,jsonify
import numpy as np
import joblib
import logging

# Application logging
# Configuring logging operations

logging.basicConfig(filename='app_deployment_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

# Create Flask object to Run
app = Flask(__name__)

# Load the model from the File

best_model = joblib.load(open("best_model.joblib","rb"))
scalar = joblib.load(open("scalar.joblib","rb"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = best_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=best_model.predict(final_input)[0]
    return render_template("index.html",prediction_text="The Insurance Premium is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)