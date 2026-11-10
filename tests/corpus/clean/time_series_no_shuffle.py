# expect: none
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("prices.csv", parse_dates=["date"])
X, y = df[["lag1"]], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
