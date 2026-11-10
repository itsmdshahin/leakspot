# expect: none
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
model = RandomForestClassifier().fit(X_res, y_res)
print(model.score(X_test, y_test))
