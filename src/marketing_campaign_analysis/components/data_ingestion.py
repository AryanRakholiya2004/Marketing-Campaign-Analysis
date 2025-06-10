import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.marketing_campaign_analysis.exception import CustomException
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.utils import read_csv_data
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    logging.info("Setting up Data Ingestion Configurations")
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')
    logging.info("Data Ingestion Configurations Set Up Completed !")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Starting Data ingestion process...")
        
    def initiate_data_ingestion(self):
        try:
            # Reading data from mysql
            logging.info("Calling 'read_sql_data()' function")
            df = read_csv_data('data/marketing_campaign.csv')

            # Making directories for train, test and raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Splitting data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test data to csv files
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)