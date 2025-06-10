import sys

# scikit-learn imports
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Project imports
from src.marketing_campaign_analysis.logger import logging
from src.marketing_campaign_analysis.exception import CustomException

class ModelEvaluation:
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        """
        Evaluates multiple machine learning models using GridSearchCV to find the best parameters and accuracy.
        Parameters:
        - X_train: Training feature set.
        - y_train: Training labels.
        - X_test: Testing feature set.
        - y_test: Testing labels.
        - models: Dictionary of model names and their corresponding scikit-learn model instances.
        - param: Dictionary of model names and their corresponding hyperparameter grids for GridSearchCV.
        Returns:
        - report: Dictionary containing model names and their corresponding accuracy scores on the test set.
        - best_params: Dictionary containing model names and their corresponding best hyperparameters found by GridSearchCV.
        - models_dict: Dictionary containing model names and their corresponding best model instances.
        """

        try:

            logging.info('Model Evaluation Started, Finding best parameters in model !')
            report = {}
            best_params = {}
            models_dict = {}

            for i in range(len(models)):
                model_name = list(models.keys())[i]
                model = models[model_name]
                parameters = param[model_name]

                gs = GridSearchCV(model, parameters, cv=3)
                gs.fit(X_train, y_train)


                best_model = gs.best_estimator_
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = test_model_score
                best_params[model_name] = gs.best_params_
                models_dict[model_name] = best_model

                print('\n=========================================\n')
                print("Evaluating -", model_name)
                print("Best Parameters: ", gs.best_params_)
                print(f"Accuracy - {test_model_score}")
                print('\n=========================================\n')
                logging.info(f"Model - {model_name} with accuracy - {test_model_score}")

            logging.info('Model Evaluation Completed, Best parameters found !')
            return report, best_params, models_dict

        except Exception as ex:
            raise CustomException(ex,sys)