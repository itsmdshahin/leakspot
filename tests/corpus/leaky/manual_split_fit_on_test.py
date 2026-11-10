# expect: LS103
from sklearn.linear_model import Ridge
X, y = load()
n = int(len(X) * 0.8)
X_train, X_test = X[:n], X[n:]
y_train, y_test = y[:n], y[n:]
m = Ridge()
m.fit(X_test, y_test)
