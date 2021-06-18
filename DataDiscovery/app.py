from flask import Flask
from flask import request
app = Flask(__name__)
#app.Config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

@app.route('/')
def home():
    return '<h1> API server is working </h1>'

@app.route('/predict')
def predict():
    return '<h1> This is the predict route </h1>'


app.run()


###################################Quick Check
# from flask import Flask
# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello World!"