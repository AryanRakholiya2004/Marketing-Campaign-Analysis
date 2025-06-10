import os
import sys
import pandas as pd
import joblib

# Project imports
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.exception import CustomException


def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    """
    Saves an object to a file using joblib.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    logging.info(f'Saving object to {file_path}')
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            joblib.dump(obj, file_obj)
        logging.info('Pickle file saved successfully !')
    except Exception as ex:
        raise CustomException(ex,sys)
