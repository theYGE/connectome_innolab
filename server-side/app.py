from flask import Flask
import random
from flask import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello_world():
    return "Initial Page"

@app.route("/makePrediction", methods=["POST"])
@cross_origin()
def model_prediction():

    #TODO: This is a temp solution
    probability = random.uniform(0,1)
    return jsonify(probability)
    # return "<h1>Under development</h1>"

    # TODO: Step 1: Get connectivity matrix
    # TODO: Step 2: Run connectivity matrix through autoencoder
    # TODO: Step 3: Get model output and return it