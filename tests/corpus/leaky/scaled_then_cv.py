# expect: LS104
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
X, y = load()
X = SelectKBest(k=20).fit_transform(X, y)
scores = cross_val_score(SVC(), X, y, cv=5)
