import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from imblearn.pipeline import Pipeline as ImPipeline 
from utils import save_obj
import os

@dataclass
class DataTransformConfig:
    preprocessor_object_train = os.path.join("/Users/siddhant/Project2/Bank_functions", "transformer.pkl")
    X_train_transformed = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "X_train_transformed.csv")
    X_train_transformed_im = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "X_train_transformed_im.csv")
    X_test_transformed = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "X_test_transformed.csv")
    Y_resampled = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "Y_resampled.csv")
    Y_test = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "Y_test.csv")
    Y_train = os.path.join("/Users/siddhant/Project2/Bank_functions/data/artifacts_module1", "Y_train_im.csv")
    
class InitiateDataTransformation:
    def __init__(self, train_file_path, test_file_path):
        self.transform_config = DataTransformConfig()
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.df1 = pd.read_csv(self.train_file_path)  
        self.df2 = pd.read_csv(self.test_file_path)   
        self.train_X, self.train_y, self.test_X, self.test_y = self.clean()
        
    
    def clean(self):
        try:
            logging.info('Creating new features for train')
            self.df1['org_balance_change'] = self.df1.apply(lambda x: 1 if (x['newbalanceOrig'] - x['oldbalanceOrg']) > 0 else 0, axis=1)
            self.df1['dest_balance'] = self.df1.apply(lambda x: abs(x['newbalanceDest'] - x['oldbalanceDest']), axis=1)
            self.df1['dest_balance_change'] = self.df1.apply(lambda x: 1 if (x['newbalanceDest'] - x['oldbalanceDest']) > 0 else 0, axis=1)
            logging.info('New features created')
            
            logging.info('Creating new features for test')
            self.df2['org_balance_change'] = self.df2.apply(lambda x: 1 if (x['newbalanceOrig'] - x['oldbalanceOrg']) > 0 else 0, axis=1)
            self.df2['dest_balance'] = self.df2.apply(lambda x: abs(x['newbalanceDest'] - x['oldbalanceDest']), axis=1)
            self.df2['dest_balance_change'] = self.df2.apply(lambda x: 1 if (x['newbalanceDest'] - x['oldbalanceDest']) > 0 else 0, axis=1)
            logging.info('New features created')
            
            logging.info("Dropping columns")
            self.df1.drop(["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 'nameOrig', 'nameDest'],axis=1,inplace=True)
            self.df2.drop(["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 'nameOrig', 'nameDest'],axis=1,inplace=True)
            logging.info("Dropped columns")
            
            logging.info('Splitting input and target variables')
            train_X = self.df1.drop(columns='isFraud')
            train_y = self.df1['isFraud']
            test_X = self.df2.drop(columns='isFraud')
            test_y = self.df2['isFraud']
            logging.info('Split successful')
            
            return train_X, train_y, test_X, test_y
            
        except Exception as e:
            logging.info('Following error occured while cleaning: {e}')
            raise CustomException(e, sys)
    
    def transform_train(self):
        try:
            encode = ['type']
            scale = ['step', 'amount', 'dest_balance']
            
            logging.info(f'Encoding training feature {encode}')
            logging.info(f'Scaling training features {scale}')
            transformer = ColumnTransformer(transformers=[
                ('ohe', OneHotEncoder(sparse_output=False), encode),
                ('scale', StandardScaler(), scale)
            ], remainder='passthrough')
            logging.info('Transformations completed')
            
            return transformer
        
        except Exception as e:
            logging.info(f'Error occured while transformation at: {e}')
            raise CustomException(e, sys)
    
    def transform_test(self):
        try:
            encode = ['type']
            scale = ['step', 'amount', 'dest_balance']

            logging.info(f'Encoding test feature {encode}')
            logging.info(f'Scaling test features {scale}')
            transformer = ColumnTransformer(transformers=[
                ('ohe', OneHotEncoder(sparse_output=False), encode),
                ('scale', StandardScaler(), scale)
            ])

            return transformer
        
        except Exception as e:
            logging.info(f'Error occured while transformation at: {e}')
            raise CustomException(e, sys)
    
    def initiate_transform(self):
        try:
            logging.info('Creating transform object for training data')
            transform_train = self.transform_train().fit(self.train_X)  
            
            logging.info('Resampling train data')
            smote = SMOTEN(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.train_X, self.train_y)
            
            logging.info('Applying transformation on training')
            X_train_transformed = transform_train.transform(X_resampled) 
            X_train_transformed_im = transform_train.transform(self.train_X) 
            
            logging.info('Creating and applying transform object for test data')
            X_test_transformed = transform_train.transform(self.test_X)

            logging.info('Saving the transformation objects')
            save_obj(file_path=self.transform_config.preprocessor_object_train, obj=transform_train)
            
            return X_train_transformed, X_train_transformed_im, y_resampled, X_test_transformed, self.test_y, self.train_y
        
        except Exception as e:
            logging.error(f'Error occurred during transformation application: {e}')
            raise CustomException(str(e), sys)
     
    def save_files(self, X_train_transformed, X_train_transformed_im, y_resampled, X_test_transformed, test_y, train_y):
        pd.DataFrame(X_train_transformed).to_csv(self.transform_config.X_train_transformed, index=False)
        pd.DataFrame(X_train_transformed_im).to_csv(self.transform_config.X_train_transformed_im, index=False)
        pd.DataFrame(y_resampled).to_csv(self.transform_config.Y_resampled, index=False)
        pd.DataFrame(X_test_transformed).to_csv(self.transform_config.X_test_transformed, index=False)
        test_y.to_csv(self.transform_config.Y_test, index=False) 
        train_y.to_csv(self.transform_config.Y_train, index=False)   
        
    def main_run(self):
        X_train_transformed, X_train_transformed_im, y_resampled, X_test_transformed, test_y, train_y = self.initiate_transform()
        self.save_files(X_train_transformed, X_train_transformed_im, y_resampled, X_test_transformed, test_y, train_y)
        

if __name__ == '__main__':
    try:
        logging.info('Transformation run')
        train_file_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/train.csv'
        test_file_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module1/test.csv'
        datatransform = InitiateDataTransformation(train_file_path, test_file_path)
        datatransform.main_run()
        logging.info('Transformation run complete')
    except Exception as e:
        logging.error(f'Error occurred in run: {e}')
        raise CustomException(str(e), sys)