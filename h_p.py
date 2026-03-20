import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100)

# Train
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# Example Prediction
sample = [[52,1,2,130,230,0,1,150,0,1.2,2,0,2]]
sample = scaler.transform(sample)

result = model.predict(sample)

if result[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
