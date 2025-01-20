from flask import Flask , request, render_template

import joblib


import numpy as np

app = Flask(__name__)

model = joblib.load("ABELY_JUSHUA_ABELY.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   try:
       Region = int(request.form['region'])
       Bedrooms = int(request.form['bedrooms']) 
       HouseType = int(request.form['house_type'])
       yearBuilt = int(request.form['year_built'])
       
       input_data = [[Region, Bedrooms, HouseType, yearBuilt]]
       
       predicted_price = model.predict(input_data)
       
       return render_template('result.html', prediction=predicted_price[0])
   except Exception as e:
       return f"Error: {str(e)}"
   
   
if __name__ == '__main__':
    app.run(debug=True)