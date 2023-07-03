from flask import Flask,render_template,url_for,request,jsonify
import requests
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder = "template")
model = pickle.load(open('model.pkl','rb'))

@app.route("/", methods = ["GET", "POST"])
def home():	

    if request.method =='POST':

        SepalLength = float(request.form.get('sepallength'))
        SepalWidth = float(request.form['sepalwidth'])
        PetalLength = float(request.form['petallength'])
        PetalWidth = float(request.form['petalwidth'])
        input_list = [SepalLength, SepalWidth, PetalLength, PetalWidth]

        pred = ' '.join(model.predict(np.array(input_list).reshape(1, -1)))
        
        return render_template("index.html", prediction = pred)
    return render_template("index.html")

if __name__ == '__main__':
	app.run(debug = True)