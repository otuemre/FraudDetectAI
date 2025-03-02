import importlib.metadata as metadata

# Define required packages
required_packages = [
    "pandas", "numpy", "matplotlib", "seaborn", "jupyter",
    "scikit-learn", "xgboost", "optuna", "imbalanced-learn", "shap"
]

def check_package(package):
    try:
        version = metadata.version(package)
        print(f"{package}: ✅ Installed (Version: {version})")
    except metadata.PackageNotFoundError:
        print(f"{package}: ❌ Not Installed! Run: pip install {package}")

def main():
    print("\nChecking required dependencies...\n")
    for package in required_packages:
        check_package(package)

if __name__ == "__main__":
    main()
