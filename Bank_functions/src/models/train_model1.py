import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from hyperopt import hp, space_eval, tpe, fmin, Trials, STATUS_OK
from dataclasses import dataclass
from utils import save_obj, open_object
import os
import mlflow


@dataclass
class ModelBuildConfig:
    model: str=os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'model.pkl')
    
class ModelBuild:
    def __init__(self, Xtrain_resampled_path, ytrain_resampled_path, Xtest_im_path, ytest_im_path):
        self.model_path = ModelBuildConfig()
        self.Xtrain_resample = pd.read_csv(Xtrain_resampled_path)
        self.Ytrain_resample = pd.read_csv(ytrain_resampled_path)
        self.X_test_transformed = pd.read_csv(Xtest_im_path)
        self.Y_test = pd.read_csv(ytest_im_path)
        logging.info(f"Xtrain_resample shape: {self.Xtrain_resample.shape}")
        logging.info(f"Ytrain_resample shape: {self.Ytrain_resample.shape}")


        # Define spaces for hyperparameter tuning
        self.xgb_space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'max_depth': hp.choice('max_depth', range(3, 10)),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
            'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
            'objective': 'binary:logistic'
        }

        self.ada_space = {
            'learning_rate': hp.uniform('ada_learning_rate', 0.01, 1.0),
            'n_estimators': hp.choice('ada_n_estimators', range(50, 1000, 50))
        }

    def incremental_objective_xgb(self, params, frac):
        try:
            X_sample, _, y_sample, _ = train_test_split(self.Xtrain_resample, self.Ytrain_resample, test_size=(1 - frac), stratify=self.Ytrain_resample)
            model = XGBClassifier(**params)
            model.fit(X_sample, y_sample)
            score = cross_val_score(model, X_sample, y_sample, scoring='roc_auc', cv=StratifiedKFold(3)).mean()
            return {'loss': -score, 'status': STATUS_OK}
        except Exception as e:
            logging.error(f'Error during XGBoost incremental objective: {e}')
            raise CustomException(e, sys)

    def incremental_objective_ada(self, params, frac):
        try:
            X_sample, _, y_sample, _ = train_test_split(self.Xtrain_resample, self.Ytrain_resample, test_size=(1 - frac), stratify=self.Ytrain_resample)
            model = AdaBoostClassifier(**params)
            model.fit(X_sample, y_sample)
            score = cross_val_score(model, X_sample, y_sample, scoring='roc_auc', cv=StratifiedKFold(3)).mean()
            return {'loss': -score, 'status': STATUS_OK}
        except Exception as e:
            logging.error(f'Error during AdaBoost incremental objective: {e}')
            raise CustomException(e, sys)

    def optimize_incrementally_xgb(self, trials, max_evals=100, increments=[0.01, 0.05, 0.1, 0.5, 1.0]):
        try:
            logging.info('Starting incremental hyperparameter tuning for XGBoost classifier with MLflow tracking.')
            with mlflow.start_run():
                best_params = None
                for frac in increments:
                    objective_func = lambda params: self.incremental_objective_xgb(params, frac)
                    best_params = fmin(fn=objective_func,
                                       space=self.xgb_space,
                                       algo=tpe.suggest,
                                       max_evals=max_evals,
                                       trials=trials)
                    mlflow.log_params(best_params)
                    mlflow.log_metric('roc_auc', -trials.best_trial['result']['loss'])
                    
                best_model_xgb = XGBClassifier(**best_params)
                best_model_xgb.fit(self.Xtrain_resample, self.Ytrain_resample)
                mlflow.xgboost.log_model(best_model_xgb, "xgb_model")
            logging.info('Completed incremental hyperparameter tuning for XGBoost.')
            return best_params
        except Exception as e:
            logging.error(f'Error during XGBoost hyperparameter tuning: {e}')
            raise CustomException(e, sys)

    def optimize_incrementally_ada(self, trials, max_evals=100, increments=[0.01, 0.05, 0.1, 0.5, 1.0]):
        try:
            logging.info('Starting incremental hyperparameter tuning for AdaBoost classifier with MLflow tracking.')
            with mlflow.start_run():
                best_params = None
                for frac in increments:
                    objective_func = lambda params: self.incremental_objective_ada(params, frac)
                    best_params = fmin(fn=objective_func,
                                       space=self.ada_space,
                                       algo=tpe.suggest,
                                       max_evals=max_evals,
                                       trials=trials)
                    mlflow.log_params(best_params)
                    mlflow.log_metric('roc_auc', -trials.best_trial['result']['loss'])
                
                best_model_ada = AdaBoostClassifier(**best_params)
                best_model_ada.fit(self.Xtrain_resample, self.Ytrain_resample)
                mlflow.sklearn.log_model(best_model_ada, "ada_model")
            logging.info('Completed incremental hyperparameter tuning for AdaBoost.')
            return best_params
        except Exception as e:
            logging.error(f'Error during AdaBoost hyperparameter tuning: {e}')
            raise CustomException(e, sys)

    def final_model(self, parameters):
        try:
            logging.info('Training final XGBoost model with optimized parameters.')
            xgb_model = XGBClassifier(**parameters)
            xgb_model.fit(self.Xtrain_resample, self.Ytrain_resample)
            logging.info('Predicting on the test set with the final model.')
            prediction = xgb_model.predict(self.X_test_transformed)
            roc_score = roc_auc_score(self.Y_test, prediction)
            recall = recall_score(self.Y_test, prediction)
            logging.info(f'Final ROC AUC score: {roc_score}')
            logging.info(f'Final RECALL score: {recall}')
            save_obj(model_builder.model_path.model, xgb_model)
            logging.info('Model saved')
            return roc_score, xgb_model
        except Exception as e:
            logging.error(f'Error in final XGBoost model training or evaluation: {e}')
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        logging.info('Initializing model building.')
        Xtrain_resampled = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/X_train_transformed.csv'
        ytrain_resampled = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/Y_resampled.csv'
        Xtest_im = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/X_test_transformed.csv'
        ytest_im = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/Y_test.csv'
        
        model_builder = ModelBuild(Xtrain_resampled, ytrain_resampled, Xtest_im, ytest_im)
        best_params_xgb = model_builder.optimize_incrementally_xgb(Trials())
        roc_score, trained_model = model_builder.final_model(best_params_xgb)
        logging.info(f'Model building completed with ROC AUC: {roc_score}')
        
    except Exception as e:
        logging.error(f'Error in model building process: {e}')
        raise CustomException(e, sys)
        
           