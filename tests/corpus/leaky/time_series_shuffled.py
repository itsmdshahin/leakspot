# expect: LS106
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("prices.csv", parse_dates=["date"])
df = df.sort_values("date")
X, y = df[["lag1", "lag2"]], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
