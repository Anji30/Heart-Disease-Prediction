import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("heart.csv")

data['target'].value_counts().plot(kind='bar')

plt.title("Heart Disease Analysis")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Count")

plt.show()


