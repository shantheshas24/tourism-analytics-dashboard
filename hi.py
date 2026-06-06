import pandas as pd

df = pd.read_csv("data/Review_db.csv")

print(df.isnull().sum())
print("\nUnique Cities:", df["City"].nunique())
print("Unique Places:", df["Place"].nunique())
print("\nRating Values:")
print(df["Rating"].value_counts().sort_index())