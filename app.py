# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:51:10 2020

@author: Admin
"""

from flask import Flask,render_template,url_for,request
import pandas as pd

import pickle

load_model = pickle.load(open('linear_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')

def home():
    
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    df = pd.read_csv('aqi.csv')
    pred = load_model.predict(df.iloc[:,:-1].values)
    pred = pred.toliat()
    return render_template('result.html',pred)


if __name__ == '__main__':
	app.run(debug=True)