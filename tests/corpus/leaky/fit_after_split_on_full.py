# expect: LS101
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
