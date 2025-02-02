import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yaml
from sklearn.preprocessing import StandardScaler

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_batch_data(df):
    """Preprocess batch data to match training features."""
    # List of features used in training
    required_features = [
        'Air_temperature', 'Process_temperature', 'Rotational_speed_rpm',
        'Torque_Nm', 'Tool_wear_min', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
        'Type_L', 'Type_M'
    ]
    
    # Drop unnecessary columns if they exist
    columns_to_drop = ['UDI', 'Product_ID', 'Machine_failure']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Process machine type if it exists
    if 'Type' in df.columns:
        df['Type_L'] = (df['Type'] == 'L').astype(int)
        df['Type_M'] = (df['Type'] == 'M').astype(int)
        df = df.drop('Type', axis=1)
    
    # Verify all required features are present
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features in the correct order
    return df[required_features]

def main():
    st.title("Machine Failure Prediction System")
    
    # Sidebar with app description
    st.sidebar.header("About")
    st.sidebar.info(
        "This application predicts machine failures based on sensor data "
        "and operational parameters. Upload your data or input values manually "
        "to get predictions."
    )
    
    # Load model
    config = load_config()
    model = load_model(config["model_path"])
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.header("Single Machine Prediction")
        
        # Input fields for all features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Operational Parameters")
            air_temp = st.number_input("Air Temperature (°C)", value=25.0, step=0.1)
            process_temp = st.number_input("Process Temperature (°C)", value=35.0, step=0.1)
            rot_speed = st.number_input("Rotational Speed (RPM)", value=1500, step=10)
            torque = st.number_input("Torque (Nm)", value=40.0, step=0.1)
            tool_wear = st.number_input("Tool Wear (min)", value=0, step=1)
        
        with col2:
            st.subheader("Failure Types")
            twf = st.checkbox("Tool Wear Failure (TWF)")
            hdf = st.checkbox("Heat Dissipation Failure (HDF)")
            pwf = st.checkbox("Power Failure (PWF)")
            osf = st.checkbox("Overstrain Failure (OSF)")
            rnf = st.checkbox("Random Failure (RNF)")
        
        with col3:
            st.subheader("Machine Type")
            machine_type = st.selectbox("Select Machine Type", options=['L', 'M', 'H'])
        
        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Air_temperature': [air_temp],
                'Process_temperature': [process_temp],
                'Rotational_speed_rpm': [rot_speed],
                'Torque_Nm': [torque],
                'Tool_wear_min': [tool_wear],
                'TWF': [1 if twf else 0],
                'HDF': [1 if hdf else 0],
                'PWF': [1 if pwf else 0],
                'OSF': [1 if osf else 0],
                'RNF': [1 if rnf else 0],
                'Type_L': [1 if machine_type == 'L' else 0],
                'Type_M': [1 if machine_type == 'M' else 0]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            # Show prediction
            st.subheader("Prediction Results")
            if prediction[0] == 1:
                st.error("⚠️ Machine Failure Predicted!")
                st.write(f"Probability of failure: {probability[0][1]:.2%}")
            else:
                st.success("✅ Machine Operating Normally")
                st.write(f"Probability of normal operation: {probability[0][0]:.2%}")
    
    with tab2:
        st.header("Batch Prediction")
        
        # Add sample CSV template
        st.info("Make sure your CSV file contains the required columns. You can keep additional columns; they will be automatically filtered.")
        
        # Create sample data for template
        sample_data = pd.DataFrame({
            'Air_temperature': [25.0],
            'Process_temperature': [35.0],
            'Rotational_speed_rpm': [1500],
            'Torque_Nm': [40.0],
            'Tool_wear_min': [0],
            'TWF': [0],
            'HDF': [0],
            'PWF': [0],
            'OSF': [0],
            'RNF': [0],
            'Type': ['L']
        })
        
        # Download template button
        csv_template = sample_data.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read and preprocess the uploaded data
                df_original = pd.read_csv(uploaded_file)
                df_processed = preprocess_batch_data(df_original.copy())
                
                # Make predictions
                predictions = model.predict(df_processed)
                probabilities = model.predict_proba(df_processed)
                
                # Add predictions to original dataframe
                results = df_original.copy()
                results['Prediction'] = predictions
                results['Failure_Probability'] = probabilities[:, 1]
                
                # Display results
                st.subheader("Prediction Results")
                st.write(f"Total samples: {len(predictions)}")
                st.write(f"Predicted failures: {sum(predictions)}")
                
                # Show detailed results
                st.dataframe(results)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
                
            except ValueError as e:
                st.error(f"Error processing file: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()