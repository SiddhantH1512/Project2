import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from dataclasses import dataclass
from utils import save_obj
import os
import mlflow
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ModelBuildConfig2:
    model_path = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'best_model_module2.pkl')

class ModelBuild:
    def __init__(self, Xtrain_path, Xtest_path, y_resampled_path, y_orig_path):
        self.modelconfig = ModelBuildConfig2()
        self.X_train, self.X_test, self.y_resampled, self.y_orig = self.load_files(Xtrain_path, Xtest_path, y_resampled_path, y_orig_path)

        self.model_dict = {
            'logistic_reg':LogisticRegression(),
            'svr':SVC(),
            'decision tree': DecisionTreeClassifier(),
            'random forest':RandomForestClassifier(),
            'extra trees': ExtraTreeClassifier(),
            'adaboost': AdaBoostClassifier(),
            'xgboost':XGBClassifier()
        }
        
        self.model_performance = {}
        self.models = {}
        
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
    
    
    def model_scorer(self, model_name, model):
        try:
            logging.info('Cross validating different models and logging F1 score')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score_f1 = cross_val_score(model, self.X_train, self.y_resampled, cv=kfolds, scoring='f1').mean()
            
            logging.info('Training model')
            model.fit(self.X_train, self.y_resampled)
            
            logging.info('Making predictions')
            prediction = model.predict(self.X_test)
            
            logging.info('Metric evaluation')
            recall = recall_score(self.y_orig, prediction)
            roc_auc = roc_auc_score(self.y_orig, prediction)
            
            logging.info('Storing model performances')
            self.model_performance[model_name] = recall
            
            logging.info('Storing models')
            self.models[model_name] = model
            
            with mlflow.start_run(run_name=model_name):
                logging.info('Logging model using mlflow')
                mlflow.sklearn.log_model(model, 'model')
                
                logging.info('Logging metrics')
                mlflow.log_metric('F1', score_f1)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('roc_auc', roc_auc)
                    
        except Exception as e:
            logging.info(f'Error in model build at: {e}')
            raise CustomException(e, sys)   
         
    def model_execution(self):
        try:
            logging.info('Execution')
            for model_name, model in self.model_dict.items():
                self.model_scorer(model_name, model) 
            
            logging.info('Extracting model with highets recall')
            best_model_perform = max(self.model_performance, key=self.model_performance.get)
            best_model = self.models[best_model_perform] 
            
            logging.info('Saving the model') 
            save_obj(self.modelconfig.model_path, best_model)
        
        except Exception as e:
            logging.info(f'Error in execution at: {e}')
            raise CustomException(e, sys)
         
    def final_run(self):
        try:
            logging.info('Final run of all methods')
            self.model_execution(self)
         
        except Exception as e:
            logging.info(f'Final run unsuccessful: {e}')
            raise CustomException(e, sys)      
            

if __name__ == '__main__':
    try:
        X_train_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/scaled_trainX.csv'
        X_test_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/scaled_testX.csv'
        y_resampled_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/resampled_Y.csv'
        y_orig_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/y_test_orig.csv'
        modelbuild = ModelBuild(X_train_path, X_test_path, y_resampled_path, y_orig_path)  
        modelbuild.model_execution()      
    
    except Exception as e:
        raise CustomException(e, sys)
    