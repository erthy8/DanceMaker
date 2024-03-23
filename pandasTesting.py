import pandas as pd

colNames = ["A" , "B"]
df = pd.DataFrame(columns = colNames)
l1 = {"A": 1 , "B": 2}
l2 = {"A" : 3,"B": 4}
df = df._append(l1, ignore_index=True)
print(df.at[0, "A"])