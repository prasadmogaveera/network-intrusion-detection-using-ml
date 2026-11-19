
import pandas as pd

for file in ["binary_train.csv", "binary_test.csv"]:
    df = pd.read_csv(file)

    print(f"\n{file}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("First 5 rows:")
    print(df.head())

    print("\nLabel distribution:")
    print(df["Label"].value_counts())
