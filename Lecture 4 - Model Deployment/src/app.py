# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:28:48 2021

@author: ratch
"""

import numpy as np
import flask
import pickle

# app
app = flask.Flask(__name__)

# load model
dm = pickle.load(open("./src/logreg.pkl","rb"))

# routes
@app.route("/")
def home():
    return """
           <body> 
           <h1>Diabetes Prediction (Demo)<h1>
           <form action="/page">
                 <input type="submit" value="Go to Page" />
            </form>
           </body>"""

@app.route("/page")
def page():
   with open("./src/page.html", 'r') as viz_file:
       return viz_file.read()

@app.route("/result", methods=["GET", "POST"])
def result():
    """Gets prediction using the HTML form"""
    if flask.request.method == "POST":
        inputs = flask.request.form
        bmi = inputs["BMI"]
        age = inputs["Age"]
        gender = inputs["Gender"]
    
    fmap = {"Male": [1],
            "Female": [0]}
    
    X_new = np.array([float(bmi)] + fmap[gender] + [int(age)]).reshape(1, -1)
    yhat = dm.predict(X_new)
    if yhat[0] == 1:
        outcome = "diabetes"
    else:
        outcome = "normal"
    prob = dm.predict_proba(X_new)
    results = """
              <body>
              <h3> Diabetes Diagnosis <h3>
              <p><h4> Patient profile </h4></p>
              <table>
              <tr>
                  <td>BMI: </td>
                  <td>""" + bmi + """</td>
              </tr>
              <tr>
                  <td>Gender: </td>
                  <td>""" + gender + """</td>
              </tr>
              <tr>
                  <td>Age: </td>
                  <td>""" + age + """</td>
              </tr>
              </table>
              <p> This patient is diagnose as """ + outcome + """ with probability """ + str(round(prob[0][1], 2)) + """
              </body>"""
              
    return results

if __name__ == '__main__':
    """Connect to Server"""
    HOST = "127.0.0.1"
    PORT = "4000"
    app.run(HOST, PORT)