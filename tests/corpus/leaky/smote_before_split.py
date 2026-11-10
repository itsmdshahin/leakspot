# expect: LS102
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X, y = load()
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, stratify=y_res)
