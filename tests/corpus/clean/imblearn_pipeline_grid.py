# expect: none
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
X, y = load()
pipe = Pipeline([("smote", SMOTE()), ("rf", RandomForestClassifier())])
grid = GridSearchCV(pipe, {"rf__n_estimators": [100, 200]}, cv=5)
grid.fit(X, y)
