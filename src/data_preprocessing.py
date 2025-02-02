import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def load_data(filepath):
    """Load raw data from CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data."""
    # Drop unnecessary columns
    df = df.drop(['UDI', 'Product_ID'], axis=1)
    
    # Encode categorical 'Type' column
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df.drop('Machine_failure', axis=1)
    y = df['Machine_failure']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test, output_folder):
    """Save processed data to disk."""
    os.makedirs(output_folder, exist_ok=True)
    X_train.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index=False)

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load raw data
    raw_data_path = config["data_paths"]["raw"]
    df = load_data(raw_data_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save processed data
    output_folder = config["data_paths"]["processed"]
    save_data(X_train, X_test, y_train, y_test, output_folder)

if __name__ == "__main__":
    main()