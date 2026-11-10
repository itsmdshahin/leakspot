# expect: LS101
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X, y = load()
selector = SelectKBest(f_classif, k=10)
X_sel = selector.fit_transform(X, y)
X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y)
clf = RandomForestClassifier().fit(X_tr, y_tr)
