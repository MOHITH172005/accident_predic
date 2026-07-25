from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "test1.pkl")
test = joblib.load(MODEL_PATH)


@app.route("/")
def hello_world():
    return render_template("t.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        int_features = [int(x) for x in request.form.values()]
        final = np.array([int_features])

        prediction = test.predict(final)

        if prediction[0] == 0:
            result = "Probability of accident severity is : Minor"
        else:
            result = "Probability of accident severity is : Major"

        return render_template("t.html", pred=result)

    return render_template("t.html")


@app.route("/Map")
def map1():
    return render_template("map.html")


@app.route("/Graphs")
def graph():
    return render_template("graph.html")


@app.route("/Map1")
def map2():
    return render_template("ur.html")


@app.route("/Map2")
def map3():
    return render_template("bs.html")


@app.route("/Map3")
def map4():
    return render_template("hm.html")


@app.route("/Pie")
def pie():
    return render_template("pie.html")


if __name__ == "__main__":
    app.run(debug=True)
