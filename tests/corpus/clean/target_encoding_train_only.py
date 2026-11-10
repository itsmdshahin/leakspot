# expect: none
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("loans.csv")
y = df["defaulted"]
train, test = train_test_split(df, test_size=0.2)
rates = train.groupby("region")["defaulted"].mean()
train["rate"] = train["region"].map(rates)
test["rate"] = test["region"].map(rates)
