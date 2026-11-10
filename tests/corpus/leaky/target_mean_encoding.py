# expect: LS105
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("loans.csv")
df["region_default_rate"] = df.groupby("region")["defaulted"].transform("mean")
X = df.drop("defaulted", axis=1)
y = df["defaulted"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
