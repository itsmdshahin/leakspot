# expect: none
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y)
m = GradientBoostingClassifier().fit(X_train, y_train)
print(m.score(X_test, y_test))
final = GradientBoostingClassifier().fit(X, y)
