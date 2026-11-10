# expect: LS101
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
df = pd.read_csv("d.csv")
imp = SimpleImputer(strategy="median")
df[["a", "b"]] = imp.fit_transform(df[["a", "b"]])
train, test = train_test_split(df, test_size=0.25)
