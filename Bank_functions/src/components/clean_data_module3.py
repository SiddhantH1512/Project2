import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from utils import save_obj
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTEN
from dataclasses import dataclass
import os

@dataclass
class CleanDataConfig:
    Xtrain_transformed: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'transformed_Xtrain.csv')
    Y_resampled: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'Y_resampled.csv')
    Xtest_transformed: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'transformed_Xtest.csv')
    Y_orig_test: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'Y_imbalanced_test.csv')
    Xtrain_transformed_im: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'transformed_im_Xtrain.csv')
    Y_orig_train: str=os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module3', 'Y_imbalanced_train.csv')
    transformer: str=os.path.join('/Users/siddhant/Project2/Bank_functions/src/models/transformer_objects', 'transformer_module3.pkl')
    
class DataTransform:
    def __init__(self, train_path, test_path):
        self.cleandata = CleanDataConfig()
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.trainX, self.trainY = self.clean(self.train)
        self.testX, self.testY = self.clean(self.test)
        
    def clean(self, dataframe):
        try:
            logging.info('Clean process started')
            logging.info('Dropping columns')
            dataframe.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Complain'], inplace=True)
            logging.info('Data Splitting')
            X = dataframe.drop(columns='Exited')
            y = dataframe['Exited']        
            
            logging.info('Clean process completed')
            return X, y

        except Exception as e:
            logging.info(f'Error occured in cleaning the data at {e}')
            raise CustomException(e, sys)
        
    def transform(self):
        try:
            ohe = OneHotEncoder(sparse_output=False)
            scaler = StandardScaler()
            
            logging.info('Creating a column transformer')
            encode_ohe = ['Geography', 'Gender', 'Card Type']
            scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned']

            transform = ColumnTransformer(transformers=[
                ('ohe', ohe, encode_ohe),
                ('scale', scaler, scale)
            ], remainder='passthrough')
            
            return transform

        except Exception as e:
            logging.info(f'Error occured in creating a column transformer the data at {e}')
            raise CustomException(e, sys)
        
    def initialise_transform(self):
        try:
            logging.info('Data transformation process begins')
            
            logging.info('Resampling the imbalanced data')
            smote = SMOTEN(sampling_strategy='auto', random_state=42)
            X_resampled3, Y_resampled3 = smote.fit_resample(self.trainX, self.trainY)
            
            logging.info('Applying column transformer of resampled train data, imbalanced train and test data')
            transformer = self.transform()
            transformer.set_output(transform='pandas')
            
            transformer.fit(X_resampled3)
            transformed_Xtrain = transformer.transform(X_resampled3)
            transformed_Xtrain_im = transformer.transform(self.trainX)
            transformed_Xtest = transformer.transform(self.testX)
            
            logging.info('Saving transformer')
            save_obj(self.cleandata.transformer, transformer)
            
            logging.info('Data transformation process complete')
            return transformed_Xtrain, transformed_Xtrain_im, Y_resampled3, transformed_Xtest
            
        except Exception as e:
            logging.info(f'Error occured in transforming the data at {e}')
            raise CustomException(e, sys)
        
    def save_files(self, transform_trainX, resampledY, transformed_testX, imbalanced_testY, transformed_im_trainX, imbalanced_trainY):
        try:
            logging.info('Saving files')
            transform_trainX.to_csv(self.cleandata.Xtrain_transformed, index=False)
            resampledY.to_csv(self.cleandata.Y_resampled, index=False)
            transformed_testX.to_csv(self.cleandata.Xtest_transformed, index=False)
            imbalanced_testY.to_csv(self.cleandata.Y_orig_test, index=False)
            transformed_im_trainX.to_csv(self.cleandata.Xtrain_transformed_im, index=False)
            imbalanced_trainY.to_csv(self.cleandata.Y_orig_train, index=False)
            
        except Exception as e:
            logging.info(f'Error occured in saving files at {e}')
            raise CustomException(e, sys)
        
    def orchestrate(self):
        try:
            logging.info('Orchestrate all the runs')
            transformed_Xtrain, transformed_Xtrain_im, Y_resampled3, transformed_Xtest = self.initialise_transform()
            self.save_files(transformed_Xtrain, Y_resampled3, transformed_Xtest, self.testY, transformed_Xtrain_im, self.trainY)
        
        except Exception as e:
            logging.info(f'Error occured in orchestrating the methods at {e}')
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        train_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/train.csv'
        test_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module3/test.csv'
        datatransform = DataTransform(train_path, test_path)
        datatransform.orchestrate()
    
    except Exception as e:
            logging.info(f'Error {e}')
            raise CustomException(e, sys) 