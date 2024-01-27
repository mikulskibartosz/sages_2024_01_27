import os
import pandas as pd


URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

os.makedirs("data", exist_ok=True)
df = pd.read_csv(URL)
df.to_csv("data/iris.csv", index=False)
