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
import os
import mlflow
import warnings

warnings.filterwarnings('ignore')

class FinalModelConfig:
    final_model = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'final_model_module2.pkl')
    
class HyperParamsTune:
    def __init__(self, obj_filepath, Xtrain_path, Xtest_path, y_resampled_path, y_orig_path):
        self.finalmodel = FinalModelConfig()
        self.xgb_model = open_object(obj_filepath)
        self.X_train, self.X_test, self.y_resampled, self.y_orig = self.load_files(Xtrain_path, Xtest_path, y_resampled_path, y_orig_path)
        
        self.xgb_space = {
            'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 650]),
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 10]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.choice('min_child_weight', [1, 5, 10]),
            'gamma': hp.uniform('gamma', 0, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0.1, 1),
        }
    
    def load_files(self, Xtrain_path, Xtest_path, y_resampled_path, y_orig_path):
        try:
            logging.info('Loading required files in dataframes')
            df_Xtrain = pd.read_csv(Xtrain_path)
            df_Xtest = pd.read_csv(Xtest_path)
            df_y_resampled = pd.read_csv(y_resampled_path)
            df_y_orig = pd.read_csv(y_orig_path)
            
            return df_Xtrain, df_Xtest, df_y_resampled, df_y_orig
        
        except Exception as e:
            logging.info(f'Error in loading files at: {e}')
            raise CustomException(e, sys)
    
    def objective_xgb(self, params):
        try:
            logging.info('Setting up the objective function')
            self.xgb_model.set_params(**params)
            
            logging.info('Performing cross validation')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score = cross_val_score(self.xgb_model, self.X_train, self.y_resampled, cv=kfolds, scoring='recall').mean()

            return {'loss': -score, 'status': STATUS_OK}
        
        except Exception as e:
            logging.info(f'Objective function error {e}')
            raise CustomException(e, sys)
        
    def optimize_xgb(self):
        try:
            trials=Trials()
            logging.info('Optimizing XGB with parameter space to get best params')
            with mlflow.start_run(run_name='Final_XGB'):
                best_xgb = fmin(fn=self.objective_xgb,
                                max_evals=100,
                                trials=trials,
                                space=self.xgb_space,
                                algo=tpe.suggest)
                
                logging.info('Extracting best parameters')
                best_params = space_eval(self.xgb_space, best_xgb)

            return best_params
        
        except Exception as e:
            logging.info('Hyperparameter tuning failed')
            raise CustomException(e, sys)
        
    def final_exection(self, params):
        try:
            logging.info('Evaluating final model with best params')
            xgb_model = XGBClassifier(**params)
            
            logging.info('Training the model')
            xgb_model.fit(self.X_train, self.y_resampled)
            
            logging.info('Making predictions and evaluating metrics')
            prediction = xgb_model.predict(self.X_test)
            recall = recall_score(self.y_orig, prediction)
            precision = precision_score(self.y_orig, prediction)
        
            with mlflow.start_run(run_name='Main_Model_XGB'):
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('precision', precision)
                mlflow.xgboost.log_model(xgb_model, 'XGB_Model')
            
            save_obj(self.finalmodel.final_model, xgb_model)

            return recall, precision
        
        except Exception as e:
            logging.info('Final execution failed')
            raise CustomException(e, sys) 
        

if __name__ == "__main__":
    try:
        logging.info('Initialise model building')
        obj_filepath = '/Users/siddhant/Project2/Bank_functions/src/models/trained_models/best_model_module2.pkl'
        Xtrain_filepath = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/scaled_trainX.csv'
        Xtest_filepath = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/scaled_testX.csv'
        y_resampled = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/resampled_Y.csv'
        y_orig = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/y_test_orig.csv'
        
        hyperparams = HyperParamsTune(obj_filepath, Xtrain_filepath, Xtest_filepath, y_resampled, y_orig)
        best_parameters = hyperparams.optimize_xgb()
        best_recall, best_precision = hyperparams.final_exection(best_parameters)
        logging.info(f'Model building completed with Recall: {best_recall} and precision {best_precision}')
        
    except Exception as e:
        logging.error(f'Error in model building process: {e}')
        raise CustomException(e, sys)
        