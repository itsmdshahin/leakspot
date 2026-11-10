# expect: LS101
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
texts, labels = load_texts()
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels)
