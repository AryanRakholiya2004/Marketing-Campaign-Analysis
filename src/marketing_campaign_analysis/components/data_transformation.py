import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Project imports
from src.marketing_campaign_analysis.exception import CustomException
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.utils import save_object

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.numerical_features: list = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'months_since_enrolled']
        self.ordinal_features: list = ['Education']
        self.nominal_features: list = ['Marital_Status']


        self.data_transformation_config = DataTransformationConfig()
        self.prepro_obj = None
        logging.info("Data Transformation Initialized")

    def feature_engineering(self, df):
        """
        This function is used to perform feature engineering on the input DataFrame.

        :param df: Input DataFrame containing the data.
        :return: DataFrame with the new features with appropriate data types.
        """
        logging.info("Feature Engineering Started")
        try:
            # converted to boolean
            df['Kidhome'] = df['Kidhome'].astype('bool')
            df['Teenhome'] = df['Teenhome'].astype('bool')
            df['AcceptedCmp1'] = df['AcceptedCmp1'].astype('bool')
            df['AcceptedCmp2'] = df['AcceptedCmp2'].astype('bool')
            df['AcceptedCmp3'] = df['AcceptedCmp3'].astype('bool')
            df['AcceptedCmp4'] = df['AcceptedCmp4'].astype('bool')
            df['AcceptedCmp5'] = df['AcceptedCmp5'].astype('bool')
            df['Complain'] = df['Complain'].astype('bool')
            df['Response'] = df['Response'].astype('bool')

            # converted to datetime
            df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

            # converted to int16
            df['Year_Birth'] = df['Year_Birth'].astype('int16')
            df['Recency'] = df['Recency'].astype('int16')
            df['MntWines'] = df['MntWines'].astype('int16')
            df['MntFruits'] = df['MntFruits'].astype('int16')
            df['MntMeatProducts'] = df['MntMeatProducts'].astype('int16')
            df['MntFishProducts'] = df['MntFishProducts'].astype('int16')
            df['MntSweetProducts'] = df['MntSweetProducts'].astype('int16')
            df['MntGoldProds'] = df['MntGoldProds'].astype('int16')
            df['NumDealsPurchases'] = df['NumDealsPurchases'].astype('int16')
            df['NumCatalogPurchases'] = df['NumCatalogPurchases'].astype('int16')
            df['NumStorePurchases'] = df['NumStorePurchases'].astype('int16')
            df['NumWebPurchases'] = df['NumWebPurchases'].astype('int16')
            df['NumWebVisitsMonth'] = df['NumWebVisitsMonth'].astype('int16')

            # converted to int8
            df['Z_CostContact'] = df['Z_CostContact'].astype('int8')
            df['Z_Revenue'] = df['Z_Revenue'].astype('int8')


            ref_date = df['Dt_Customer'].max()
            df['months_since_enrolled'] = (ref_date - df['Dt_Customer']).dt.days // 30

            logging.info("Feature Engineering Completed Successfully")
            return df
        except Exception as ex:
            raise CustomException(ex, sys)
    

    def get_data_tranformation_instance(self):
        '''
        This function is used to create a data transformation pipeline.

        returns:
        preprocessor: ColumnTransformer object that contains the numerical and categorical pipelines.
        '''
        logging.info("Data Transformation Started")


        try:

            logging.info("Defining numerical and categorical features")

            logging.info(f"Numerical features: {self.numerical_features}")
            logging.info(f"Ordinal features: {self.ordinal_features}")
            logging.info(f"Nominal features: {self.nominal_features}")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            ord_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OrdinalEncoder()),
            ])

            nom_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer(
                [
                    ('numerical', num_pipeline, self.numerical_features),
                    ('ordinal', ord_pipeline, self.ordinal_features),
                    ('nominal', nom_pipeline, self.nominal_features)
                ]
            )

            logging.info("Data Transformation pipeline created successfully")
            return preprocessor

        except Exception as ex:
            raise CustomException(ex, sys)
        


    def initiate_data_transformation(self,train_path, test_path):
        """
        This function is used to initiate the data transformation process.
        It reads the train and test data, applies the preprocessing steps,
        and saves the preprocessor object to a file.

        :param train_path: Path to the training data CSV file.
        :param test_path: Path to the testing data CSV file.

        :return: Tuple containing the transformed training and testing data arrays,
                 and the path to the saved preprocessor object.
        """
        try:
            logging.info("Called 'initiate_data_transformation' function")

            # Read the data
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path,sep='\t')
            test_df = pd.read_csv(test_path,sep='\t')
            logging.info("Data read successfully from the Train/Test CSV files")

            # Feature engineering
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            preprocessing_obj = self.get_data_tranformation_instance()

                # Separate features and target variable
            logging.info("Separating features and target variable")
            target_column_name = 'Response'
                # Trainig set
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
                # Testing set
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            logging.info("Features and target variable separated successfully")

            x_train_arr = preprocessing_obj.fit_transform(X_train)
            x_test_arr = preprocessing_obj.transform(X_test)

            print('='*36)
            print('\nPreprocessor information - before return from function')
            print(preprocessing_obj)
            print(type(preprocessing_obj))
            print('='*36)

            train_arr = np.c_[x_train_arr, np.array(y_train)]
            test_arr = np.c_[x_test_arr, np.array(y_test)]
            logging.info("Data Transformation completed successfully")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as ex:
            raise CustomException(ex, sys)
