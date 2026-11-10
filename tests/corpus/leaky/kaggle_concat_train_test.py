# expect: LS103
import pandas as pd
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test])
full[["age"]] = StandardScaler().fit_transform(full[["age"]])
