# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from ML.model.segment import Edge_detection
import datetime
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def hello():
    return "hello"

@app.route('/run_model')
def run_model():
    Edge_detection()
    return "successful"

'''
@app.route('/predict_bike_sale')
def predict_sale():
    bike_rental(Field1_name, Field2_name)

'''


# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=0)
