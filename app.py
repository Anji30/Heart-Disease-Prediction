from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = round(model.score(X_test, y_test) * 100, 2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    features = [float(request.form[x]) for x in request.form if x != "name"]

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1] * 100

    result = "No Disease" if prediction == 0 else "Disease Detected"
    color = "green" if prediction == 0 else "red"

    return render_template("result.html",
                           name=name,
                           result=result,
                           probability=round(probability, 2),
                           accuracy=accuracy,
                           color=color)


if __name__ == "__main__":
    app.run(debug=True)



