import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTEN
from dataclasses import dataclass
import os

@dataclass
class DataTransformconfig:
    scaled_X_train = os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module2', 'scaled_trainX.csv')
    scaled_X_test = os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module2', 'scaled_testX.csv')
    resampled_y = os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module2', 'resampled_Y.csv')
    orig_y_test = os.path.join('/Users/siddhant/Project2/Bank_functions/data/artifacts_module2', 'y_test_orig.csv')

class DataTransform:
    def __init__(self, train_file_path, test_file_path):
        self.dataconfig = DataTransformconfig()
        self.traindf = pd.read_csv(train_file_path)
        self.testdf = pd.read_csv(test_file_path)
        self.train_X2, self.train_Y2 = self.clean(self.traindf)
        self.test_X2, self.test_Y2 = self.clean(self.testdf, is_train=False)
        
    def clean(self, df, is_train=True):
        try:
            logging.info('deleting unwanted Column from train and test')
            df.drop(columns=['ID', 'Experience'], inplace=True, axis=1)
            
            logging.info('Column cleanup')
            df['CCAvg'] = df['CCAvg'] * 12

            if is_train:
                logging.info('Setting upper limit to catch outliers')
                upper_limit_train = df['Mortgage'].mean() + 3*df['Mortgage'].std()
        
                logging.info('Removing outlier')
                df = df[~(df['ZIP Code'] < 20000)]
                df = df[~(df['Mortgage'] > upper_limit_train)]
            
            logging.info('Splitting input and target variables')
            X = df.drop(columns='Personal Loan')
            y = df['Personal Loan']
            
            return X, y
            
        except Exception as e:
            logging.info(f'Error occured in cleaning at: {e}')
            raise CustomException(e, sys) 

    def transform(self):
        try:
            logging.info('Resampling the data')
            smote = SMOTEN(sampling_strategy='auto', random_state=42)
            resampled_Xtrain2, resampled_Ytrain2 = smote.fit_resample(self.train_X2, self.train_Y2)
            
            logging.info('Scaling the data')
            scaler = StandardScaler()
            scaler.set_output(transform='pandas')
            scaled_Xtrain2 = scaler.fit_transform(resampled_Xtrain2)
            scaled_Xtest2 = scaler.transform(self.test_X2)
            
            logging.info('Dropping low importance columns')
            scaled_Xtrain2.drop(columns=['Mortgage', 'Securities Account', 'Online', 'CreditCard'], inplace=True)
            scaled_Xtest2.drop(columns=['Mortgage', 'Securities Account', 'Online', 'CreditCard'], inplace=True)
            
            self.save_files(scaled_Xtrain2, resampled_Ytrain2, scaled_Xtest2, self.test_Y2)
        
        except Exception as e:
            logging.info(f'Error occured in data transformation at: {e}')
            raise CustomException(e, sys) 

    def save_files(self, Xtrain_scaled, y_resampled, Xtest_scaled, imbalanced_y):
        logging.info('Saving files')
        Xtrain_scaled.to_csv(datatransform.dataconfig.scaled_X_train, index=False)
        Xtest_scaled.to_csv(datatransform.dataconfig.scaled_X_test, index=False)
        y_resampled.to_csv(datatransform.dataconfig.resampled_y, index=False)
        imbalanced_y.to_csv(datatransform.dataconfig.orig_y_test, index=False)

    def orchestrate(self):
        self.transform()
    
if __name__ == '__main__':
    try:
        logging.info('Running the script')
        train_file_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/train.csv'
        test_file_path = '/Users/siddhant/Project2/Bank_functions/data/artifacts_module2/test.csv'
        datatransform = DataTransform(train_file_path, test_file_path)
        datatransform.orchestrate()
    except Exception as e:
            logging.info(f'Error occured in running the script at: {e}')
            raise CustomException(e, sys) 
