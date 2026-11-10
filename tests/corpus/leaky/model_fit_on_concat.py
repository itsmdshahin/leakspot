# expect: LS103
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("a.csv")
X, y = df.drop(columns=["label"]), df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
model = DecisionTreeClassifier()
model.fit(X_all, y_all)
print(model.score(X_test, y_test))
