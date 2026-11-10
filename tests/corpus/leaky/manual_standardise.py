# expect: LS101
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("x.csv")
df["income"] = (df["income"] - df["income"].mean()) / df["income"].std()
df["age"] = df["age"].fillna(df["age"].median())
train, test = train_test_split(df, test_size=0.2)
