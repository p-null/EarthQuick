from flask import request
from flask import Flask
import json
from sklearn.externals import joblib
from waitress import serve
# from script_folder import train

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/do_post", methods=['POST'])
def post_method():
    json = request.get_json()
    name = json["name"]
    age = json["age"]
    return "Hello {}. You're {} years old.".format(name, age)


@app.route("/train_model", methods=['GET'])
def train_model():
    return train.train()

@app.route("/predict", methods=['POST'])
def predict():
    json = request.get_json()
    # expect X in format [[1, 2, 4, 5]]
    X = json["X"]
    model = joblib.load("xgb_model.pkl")
    return str(model.predict(X))


# This is important so that the server will run when the docker container has been started. Host=0.0.0.0 needs to be provided to make the server publicly available.
if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
