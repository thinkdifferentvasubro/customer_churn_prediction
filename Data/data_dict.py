import pandas as pd

df = pd.read_csv(r"C:\projects\customer churn prediction\Data\churn.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.drop(["churn", "customerid"], axis=1)

row_dict = df.iloc[1].to_dict()
print(row_dict)
