import pandas as pd
df = pd.read_csv("data/raw/city_day.csv", parse_dates=["Date"])
print(df.shape)                      # expect (29531, 16)
print(df["City"].nunique())          # expect 26
print(df["Date"].min(), df["Date"].max())   # 2015-01-01 to 2020-07-01
print(df["AQI"].isna().mean())       # expect about 0.159