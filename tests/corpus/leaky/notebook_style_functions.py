# expect: LS101
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
def prepare(X):
    scaler = RobustScaler()
    return scaler.fit_transform(X)
X, y = load()
X = RobustScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
