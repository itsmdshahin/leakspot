# expect: none
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
