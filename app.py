from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, pred) * 100, 2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]

    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    if prediction == 1:
        result = "Heart Disease Detected"
        color = "text-danger"
    else:
        result = "No Heart Disease"
        color = "text-success"

    return render_template("index.html",
        prediction_text=result,
        color=color,
        accuracy=accuracy,
        probability=round(probability*100,2),
        name=name,
        age=request.form["age"],
        sex=request.form["sex"],
        cp=request.form["cp"],
        trestbps=request.form["trestbps"],
        chol=request.form["chol"]
    )

if __name__ == "__main__":
    app.run(debug=True)



