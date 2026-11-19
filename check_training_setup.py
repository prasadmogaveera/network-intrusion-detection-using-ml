
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).resolve().parent

required_files = [
    "binary_train.csv",
    "binary_test.csv",
    "multiclass_train.csv",
    "multiclass_test.csv",
]

required_packages = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "tensorflow",
    "matplotlib",
]

print("=" * 55)
print("TRAINING SETUP CHECK")
print("=" * 55)

print("\nDataset files:")
for filename in required_files:
    path = BASE_DIR / filename
    print(f"{filename}: {'FOUND' if path.exists() else 'MISSING'}")

print("\nPython packages:")
for package in required_packages:
    installed = importlib.util.find_spec(package) is not None
    print(f"{package}: {'INSTALLED' if installed else 'NOT INSTALLED'}")

print("\nSetup check complete. No models were trained.")
