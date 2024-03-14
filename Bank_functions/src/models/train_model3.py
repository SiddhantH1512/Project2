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
from sklearn.metrics import recall_score, precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from dataclasses import dataclass
from utils import save_obj
import os
import mlflow
import warnings

warnings.filterwarnings('ignore')

class ModelConfig3:
    resampled_object = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'resampled_bestmodel.pkl')
    imbalanced_object = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'imbalanced_bestmodel.pkl')
    resampled_prediction = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'resampled_prediction.pkl')
    imbalanced_prediction = os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/trained_models', 'imbalanced_prediction.pkl')

class ModelBuild:
    def __init__(self, Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path):
        self.modelconfig = ModelConfig3()
        self.Xtrain, self.Ytrain, self.Xtrain_im, self.Ytrain_im, self.Xtest, self.Ytest = self.load_files(Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path)
        
        self.model_dict = {
            'logistic_reg':LogisticRegression(),
            'svc':SVC(),
            'decision tree': DecisionTreeClassifier(),
            'random forest':RandomForestClassifier(),
            'extra trees': ExtraTreeClassifier(),
            'adaboost': AdaBoostClassifier(),
            'xgboost':XGBClassifier()
        }
        
        self.model_preformance_resampled = {}
        self.model_preformance_imbalanced = {}
        self.model_resampled = {}
        self.model_imbalanced = {}
        
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
    
    def model_scorer_resampled(self, model_name, model):
        try:
            logging.info('Cross validating different models and logging F1 score for resampled data')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score_resampled = cross_val_score(model, self.Xtrain, self.Ytrain, cv=kfolds, scoring='f1').mean()
            
            logging.info('Making predictions and calculating metrics for resampled data')
            model.fit(self.Xtrain, self.Ytrain)
            prediction_resampled = model.predict(self.Xtest)
            recall_resampled = recall_score(prediction_resampled, self.Ytest)
            precision_resampled = precision_score(prediction_resampled, self.Ytest)
            
            self.model_preformance_resampled[model_name] = recall_resampled
            self.model_resampled[model_name] = model
                   
            with mlflow.start_run(run_name='resample_'+ model_name):
                logging.info('Logging model and metrics')
                mlflow.log_metric('F1_resampled', score_resampled)
                mlflow.log_metric('recall_resampled', recall_resampled)
                mlflow.log_metric('precision_resampled', precision_resampled)
                mlflow.sklearn.log_model(model, 'model_name')
            
            logging.info('Saving the prediction object')
            save_obj(self.modelconfig.resampled_prediction, prediction_resampled)
            
        except Exception as e:
            logging.info(f'Error in cross validation at: {e}')
            raise CustomException(e, sys)   
            
    def model_scorer_imbalanced(self, model_name, model):
        try:
            logging.info('Cross validating different models and logging F1 score for imbalanced data')
            kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
            score_imbalanced = cross_val_score(model, self.Xtrain_im, self.Ytrain_im, cv=kfolds, scoring='f1').mean()
            
            logging.info('Making predictions and calculating metrics for imbalanced data')
            model.fit(self.Xtrain_im, self.Ytrain_im)
            prediction_imbalanced = model.predict(self.Xtest)
            recall_imbalanced = recall_score(prediction_imbalanced, self.Ytest)
            precision_imbalanced = precision_score(prediction_imbalanced, self.Ytest)
            
            self.model_preformance_imbalanced[model_name] = recall_imbalanced
            self.model_imbalanced[model_name] = model
                   
            with mlflow.start_run(run_name='imbalanced_'+ model_name):
                logging.info('Logging model and metrics')
                mlflow.log_metric('F1_imbalanced', score_imbalanced)
                mlflow.log_metric('recall_imbalanced', recall_imbalanced)
                mlflow.log_metric('precision_imbalanced', precision_imbalanced)
                mlflow.sklearn.log_model(model, 'model_name')
            
            logging.info('Saving the prediction object')
            save_obj(self.modelconfig.imbalanced_prediction, prediction_imbalanced)
            
        except Exception as e:
            logging.info(f'Error in cross validation at: {e}')
            raise CustomException(e, sys)         
            
    def model_evaluation_resampled(self):
        try:
            logging.info('Execution of all the models begin')
            for model_name, model in self.model_dict.items():
                self.model_scorer_resampled(model_name, model)

            logging.info('Extracting model with highest recall')
            best_model_name = max(self.model_preformance_resampled, key=self.model_preformance_resampled.get)
            best_model_resampled = self.model_resampled[best_model_name]
            
            logging.info('Saving the best model trained on resampled data')
            save_obj(self.modelconfig.resampled_object, best_model_resampled)
            
        except Exception as e:
            logging.info(f'Error in execution at: {e}')
            raise CustomException(e, sys)    
            
    def model_evaluation_imbalanced(self):
        try:
            logging.info('Execution of all the models begin')
            for model_name, model in self.model_dict.items():
                self.model_scorer_imbalanced(model_name, model)

            logging.info('Extracting model with highest recall')
            best_model_name_im = max(self.model_preformance_resampled, key=self.model_preformance_resampled.get)
            best_model_im = self.model_imbalanced[best_model_name_im]
            
            logging.info('Saving the best model trained on resampled data')
            save_obj(self.modelconfig.imbalanced_object, best_model_im)
            
        except Exception as e:
            logging.info(f'Error in execution at: {e}')
            raise CustomException(e, sys)          
    
if __name__ == '__main__':
    try:
        Xtrain_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_Xtrain.csv'
        y_resampled_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_resampled.csv'
        Xtrain_im_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_im_Xtrain.csv'
        Ytrain_imbalanced_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_imbalanced_train.csv'
        Xtest_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/transformed_Xtest.csv'
        y_orig_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/Y_imbalanced_test.csv'
        
        modelbuild = ModelBuild(Xtrain_path, y_resampled_path, Xtrain_im_path, Ytrain_imbalanced_path, Xtest_path, y_orig_path)  
        modelbuild.model_evaluation_resampled()
        modelbuild.model_evaluation_imbalanced()     
    
    except Exception as e:
        raise CustomException(e, sys)         
            
            
            
            
          
            
            