# expect: LS103
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
