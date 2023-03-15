"""
Backend of the project
"""
from flask import Flask
import random
from flask import jsonify
from flask_cors import CORS, cross_origin

# Initialization of the Flask App for the API calls
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello_world():
    """
    Landing page of the project
    @return: string
    """
    return "Initial Page"

@app.route("/makePrediction", methods=["POST"])
@cross_origin()
def model_prediction():
    """
    Main API method for handling API calls
    @return: int
    """
    # Generates a demo value for the prototype
    probability = random.uniform(0,1)
    return jsonify(probability)
