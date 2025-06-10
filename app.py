import sys
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.exception import CustomException

# Components imports
from src.marketing_campaign_analysis.components.data_ingestion import DataIngestion
from src.marketing_campaign_analysis.components.data_transformation import DataTransformation
from src.marketing_campaign_analysis.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Starting the Marketing Campaign Analysis Pipeline...")
    try:
        # Data ingestion process
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data transformation process
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # Model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))

    except Exception as e:
        raise CustomException(e, sys)
