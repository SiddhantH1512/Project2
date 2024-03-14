import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from sklearn.metrics import recall_score, precision_score
from xgboost import XGBClassifier
from utils import save_obj, open_object
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from sklearn.model_selection import cross_val_score, KFold
from dataclasses import dataclass
from utils import save_obj, open_object
import os
import mlflow
import warnings

warnings.filterwarnings('ignore')

@dataclass
class FinalModelConfig:
    final_model_resampled = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'final_model_resampled_module3.pkl')
    final_model_imbalanced = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'final_model_imbalanced_module3.pkl')
    
class HyperParamsTuning:
    def __init__(self, Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path):
        self.finalmodel = FinalModelConfig()
        self.xgb_model = XGBClassifier()
        self.Xtrain, self.y_resampled, self.Xtrain_im, self.y_train_im, self.Xtest, self.y_test = self.load_files(Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path)
        
        self.xgb_space = {
            'n_estimators': hp.choice('xgb_n_estimators', [100, 300, 500, 700]),
            'max_depth': hp.choice('xgb_max_depth', list(range(1, 11))),
            'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('xgb_subsample', 0.5, 0.9),
            'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 0.9),
            'min_child_weight': hp.choice('xgb_min_child_weight', [1, 5, 10]),
            'gamma': hp.uniform('xgb_gamma', 0, 1),
            'reg_alpha': hp.uniform('xgb_reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('xgb_reg_lambda', 0.001, 1)
        }
            
    def load_files(self, Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path):
        try:
            logging.info('Loading required files in dataframes')
            df_Xtrain = pd.read_csv(Xtrain_path)
            df_y_resampled = pd.read_csv(y_resampled_path)
            df_Xtrain_im = pd.read_csv(Xtrain_im_path)
            y_imbalanced_df = pd.read_csv(Ytrain_imbalanced_path)
            df_Xtest = pd.read_csv(Xtest_path)
            df_ytest_orig = pd.read_csv(y_orig_path)
            
            logging.info('Load coplete')
            return df_Xtrain, df_y_resampled, df_Xtrain_im, y_imbalanced_df, df_Xtest, df_ytest_orig
        
        except Exception as e:
            logging.info(f'Error in loading files at: {e}')
            raise CustomException(e, sys)
        
    def objective_resampled(self, params):
        try:
            logging.info('Setting up the objective function')
            self.xgb_model.set_params(**params)
            
            logging.info('Cross validating the data')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score_resample = cross_val_score(self.xgb_model, self.Xtrain, self.y_resampled, cv=kfolds, scoring='recall').mean()
            
            return{'loss': -score_resample, 'status': STATUS_OK}
        
        except Exception as e:
            logging.info(f'Cross validation error {e}')
            raise CustomException(e, sys)
    
    def objective_imbalanced(self, params):
        try:
            logging.info('Setting up the objective function')
            self.xgb_model.set_params(**params)
            
            logging.info('Cross validating the data')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score_imbalanced = cross_val_score(self.xgb_model, self.Xtrain_im, self.y_train_im, cv=kfolds, scoring='recall').mean()
            
            return{'loss': -score_imbalanced, 'status': STATUS_OK}
        
        except Exception as e:
            logging.info(f'Cross validation error {e}')
            raise CustomException(e, sys)
        
    def optimize_resampled_objective(self):
        try:
            trials = Trials()
            logging.info('Hyperparamter tuning started')
            with mlflow.start_run(run_name='final_resample_XGB'):
                best_resample_xgb = fmin(fn=self.objective_resampled,
                                         trials=trials,
                                         max_evals=100,
                                         algo=tpe.suggest,
                                         space=self.xgb_space)
                
                logging.info('Extracting best parameters')
                best_params_resampled = space_eval(self.xgb_space, best_resample_xgb)
                
            return best_params_resampled
        
        except Exception as e:
            logging.info('Hyperparameter tuning failed')
            raise CustomException(e, sys)
        
    def optimize_imbalanced_objective(self):
        try:
            trials = Trials()
            logging.info('Hyperparamter tuning started')
            with mlflow.start_run(run_name='final_imbalance_XGB'):
                best_imbalance_xgb = fmin(fn=self.objective_imbalanced,
                                         trials=trials,
                                         max_evals=100,
                                         algo=tpe.suggest,
                                         space=self.xgb_space)
                
                logging.info('Extracting best parameters')
                best_params_imbalanced = space_eval(self.xgb_space, best_imbalance_xgb)
                
            return best_params_imbalanced
        
        except Exception as e:
            logging.info('Hyperparameter tuning failed')
            raise CustomException(e, sys)
        
    def final_execution_resampled(self, params):
        try:
            logging.info('Final execution started')
            xgb_model_resampled = XGBClassifier(**params)
            
            logging.info('Training the model and making predicitions')
            xgb_model_resampled.fit(self.Xtrain, self.y_resampled)
            prediction_resampled = xgb_model_resampled.predict(self.Xtest)
            
            logging.info('Calculating metrics')
            recall_resampled = recall_score(self.y_test, prediction_resampled)
            precision_resampled = precision_score(self.y_test, prediction_resampled)
            
            with mlflow.start_run(run_name='Final_XGB_resampled'):
                mlflow.log_metric('recall_resampled', recall_resampled)
                mlflow.log_metric('precision_resampled', precision_resampled)
                mlflow.xgboost.log_model(xgb_model_resampled, 'xgb_model')
            
            logging.info('Saving the trained model')
            save_obj(self.finalmodel.final_model_resampled, xgb_model_resampled)
                
            return recall_resampled, precision_resampled
            
        except Exception as e:
            logging.info('Final execution failed')
            raise CustomException(e, sys) 
        
    def final_execution_imbalanced(self, params):
        try:
            logging.info('Final execution started')
            xgb_model_imbalanced = XGBClassifier(**params)
            
            logging.info('Training the model and making predicitions')
            xgb_model_imbalanced.fit(self.Xtrain_im, self.y_train_im)
            prediction_imbalanced = xgb_model_imbalanced.predict(self.Xtest)
            
            logging.info('Calculating metrics')
            recall_imbalanced = recall_score(self.y_test, prediction_imbalanced)
            precision_imbalanced = precision_score(self.y_test, prediction_imbalanced)
            
            with mlflow.start_run(run_name='Final_XGB_imbalanced'):
                mlflow.log_metric('recall_imbalanced', recall_imbalanced)
                mlflow.log_metric('precision_imbalanced', precision_imbalanced)
                mlflow.xgboost.log_model(xgb_model_imbalanced, 'xgb_model')
                
            logging.info('Saving the trained model')
            save_obj(self.finalmodel.final_model_imbalanced, xgb_model_imbalanced)
                
            return recall_imbalanced, precision_imbalanced
        
        except Exception as e:
            logging.info('Final execution failed')
            raise CustomException(e, sys) 

if __name__ == "__main__":
    try:
        logging.info('Initialise model building')
        Xtrain_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_Xtrain.csv'
        y_resampled_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_resampled.csv'
        Xtrain_im_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_im_Xtrain.csv'
        Ytrain_imbalanced_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_imbalanced_train.csv'
        Xtest_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_Xtest.csv'
        y_orig_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_imbalanced_test.csv'
        
        hyperparams = HyperParamsTuning(Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path)
        best_parameters_resampled = hyperparams.optimize_resampled_objective()
        best_parameters_imbalanced = hyperparams.optimize_imbalanced_objective()
        
        best_recall_resampled, best_precision_resampled = hyperparams.final_execution_resampled(best_parameters_resampled)
        best_recall_imbalanced, best_precision_imbalanced = hyperparams.final_execution_imbalanced(best_parameters_imbalanced)
        logging.info(f'Model building completed with resampled data Recall: {best_recall_resampled} and precision {best_precision_resampled}')
        logging.info(f'Model building completed with imbalanced data Recall: {best_recall_imbalanced} and precision {best_precision_imbalanced}')
        
    except Exception as e:
        logging.error(f'Error in model building process: {e}')
        raise CustomException(e, sys)
    
    
    
    