# expect: LS104
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
X, y = load()
X_res, y_res = SMOTE().fit_resample(X, y)
grid = GridSearchCV(RandomForestClassifier(), {"n_estimators": [100, 200]}, cv=5)
grid.fit(X_res, y_res)
