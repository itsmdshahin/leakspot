# expect: LS101
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
X, y = get_data()
pca = PCA(n_components=5)
pca.fit(X)
Z = pca.transform(X)
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, random_state=0)
