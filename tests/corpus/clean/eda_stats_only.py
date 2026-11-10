# expect: none
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("x.csv")
print(df["income"].mean(), df["income"].std())
summary = df.describe()
train, test = train_test_split(df, test_size=0.2)
