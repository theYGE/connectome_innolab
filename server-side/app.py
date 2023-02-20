from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Initial Page"

@app.route("/makePrediction", methods=["POST"])
def model_prediction():
    return "<h1>Under development</h1>"

    # TODO: Step 1: Get connectivity matrix
    # TODO: Step 2: Run connectivity matrix through autoencoder
    # TODO: Step 3: Get model output and return it