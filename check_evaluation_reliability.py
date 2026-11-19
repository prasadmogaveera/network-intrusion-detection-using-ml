
import pandas as pd

FILES = {
    "Binary Train": "binary_train.csv",
    "Binary Test": "binary_test.csv",
    "Multiclass Train": "multiclass_train.csv",
    "Multiclass Test": "multiclass_test.csv",
}

for name, filename in FILES.items():
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    try:
        df = pd.read_csv(filename)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        if "Label" in df.columns:
            print("\nLabel distribution:")
            print(df["Label"].value_counts())

        print("\nMissing values:", int(df.isna().sum().sum()))

    except FileNotFoundError:
        print("File not found:", filename)

print("\nEvaluation check completed.")
