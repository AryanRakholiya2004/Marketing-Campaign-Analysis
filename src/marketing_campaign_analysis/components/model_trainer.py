import os
import sys
from dataclasses import dataclass
import mlflow
import dagshub
from urllib.parse import urlparse

# Project imports
from src.marketing_campaign_analysis.exception import CustomException
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.utils import save_object
from src.marketing_campaign_analysis.components.model_evaluation import ModelEvaluation

# scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Dagshub integration
dagshub.init(repo_owner='AryanRakholiya2004',repo_name='Marketing-Campaign-Analysis', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/AryanRakholiya2004/Marketing-Campaign-Analysis.mlflow')
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info('Model Training Started !')

    def eval_matrics(self, y_test, y_pred):
        cr = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return cr, cm, acc

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForestClassifier': RandomForestClassifier(),
                'SVC': SVC(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'VotingClassifier': VotingClassifier(estimators=[
                    ('rf', RandomForestClassifier()),
                    ('dt', DecisionTreeClassifier()),
                    ('knn', KNeighborsClassifier())
                ]),
                'StackingClassifier': StackingClassifier(estimators=[
                    ('rf', RandomForestClassifier()),
                    ('dt', DecisionTreeClassifier()),
                    ('knn', KNeighborsClassifier())
                ], final_estimator=LogisticRegression()),
                'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            params = {
                'LogisticRegression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': [100, 200, 500]
                },
                'RandomForestClassifier': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy']
                },
                'SVC': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                },
                'DecisionTreeClassifier': {
                    'max_depth': range(4, 12, 2),
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [5, 10, 20, 50, 100],
                    'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                },
                'XGBClassifier': {
                    'n_estimators': [10, 20, 50, 100],
                    'learning_rate': [0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                },
                'VotingClassifier': {},
                'StackingClassifier': {}
            }

            # Model Evaluation
            model_eval = ModelEvaluation()

            score_report, params_report, models_report = model_eval.evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f'Model Evaluation Completed !{score_report}')
            print(f'\n=====================================\n MODELS REPORT -{score_report}\n=====================================\n')
            print(f'\n=====================================\nPARAMETERS REPORT -{params_report}\n=====================================\n')

            # Saving the best model
            best_model_score = max(sorted(score_report.values()))
            logging.info('Found best scores of models')

            # Finding best model name and parameters
            best_model_name = max(score_report, key=score_report.get)
            print(f'\n=====================================\nBEST MODEL NAME -{best_model_name}\n=====================================\n')
            best_model_params = params_report[best_model_name]
            print(f'\n=====================================\nBEST MODEL PARAMETERS -{best_model_params}\n=====================================\n')

            best_model = models_report[best_model_name]
            predicted = best_model.predict(X_test)
            logging.info('Found best scored model') 

            if best_model_score<0.6:
                logging.info('No best model found !')
                raise CustomException('No best model found !', sys)
    
            # Mlflow integraring
            with mlflow.start_run():
                (_, _, acc) = self.eval_matrics(y_test, predicted)
                mlflow.log_param("model_name", best_model_name)
                mlflow.log_param("Best_parameters",best_model_params)
                mlflow.log_metric("accuracy", acc)

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name="Bank-Customer-Churn-Model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            accuracy = accuracy_score(y_test, predicted)
            logging.info(f'Model evaluation completed, Model - {best_model_name} found with accuracy of {accuracy} !')
            return accuracy
            
        except Exception as ex:
            raise CustomException(ex, sys)
        
        