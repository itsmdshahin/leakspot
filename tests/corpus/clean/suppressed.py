# expect: none
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = load()
X = StandardScaler().fit_transform(X)  # leakspot: ignore[LS101]
X_train, X_test, y_train, y_test = train_test_split(X, y)
