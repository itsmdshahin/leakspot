# expect: none
import xgboost as xgb
from sklearn.model_selection import train_test_split
X, y = load()
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2)
model = xgb.XGBClassifier(early_stopping_rounds=20)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(model.score(X_test, y_test))
