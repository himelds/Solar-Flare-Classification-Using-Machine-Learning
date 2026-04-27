import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    print("Starting Training Pipeline")
    
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, feature_names = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    # Save feature names for the Streamlit app
    import pickle
    with open('artifacts/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
        
    print("Training Pipeline Completed Successfully!")
