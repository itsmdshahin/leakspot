# expect: none
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=10).fit(X_train)
clf = LogisticRegression().fit(pca.transform(X_train), y_train)
acc = clf.score(pca.transform(X_test), y_test)
