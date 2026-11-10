# expect: none
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
X, labels = load()
y = LabelEncoder().fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y)
