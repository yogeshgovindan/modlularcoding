import os

# Define folders to create
folders = [
    "data/raw",
    "data/processed",
    "data/splits",
    "models",
    "notebooks",
    "src",
    "app/templates"
]

# Define files to create
files = [
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/utils.py",
    "app/__init__.py",
    "app/app.py",
    "requirements.txt",
    "config.yaml",
    "main.py"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create files
for file in files:
    with open(file, "w") as f:
        pass
    print(f"Created file: {file}")

print("Folder and file structure created successfully!")