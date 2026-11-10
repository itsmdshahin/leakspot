# expect: LS101
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("titanic.csv")
y = df["Survived"]
X = df.drop("Survived", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train["Fare"] = X_train["Fare"].fillna(np.mean(X["Fare"]))
