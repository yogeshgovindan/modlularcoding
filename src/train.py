import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn
import yaml
import os  # Add this import statement

def load_processed_data(processed_folder):
    """Load processed data from disk."""
    X_train = pd.read_csv(os.path.join(processed_folder, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_folder, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_folder, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(processed_folder, "y_test.csv"))
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    return accuracy

def save_model(model, model_path):
    """Save the trained model to disk."""
    joblib.dump(model, model_path)

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    processed_folder = config["data_paths"]["processed"]
    X_train, X_test, y_train, y_test = load_processed_data(processed_folder)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Log metrics with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    # Save model
    model_path = config["model_path"]
    save_model(model, model_path)

if __name__ == "__main__":
    main()