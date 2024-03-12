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

class DataTransformconfig:
    pass

class DataTransform:
    def __init__(self, train_file_path, test_file_path):
        self.dataconfig = DataTransformconfig()
        self.traindf = pd.read_csv(train_file_path)
        self.testdf = pd.read_csv(test_file_path)
        self.train_X2, self.train_y2, self.test_X2, self.test_y2 = self.clean()
        
    def clean(self):
        try:
            for df in zip(self.traindf, self.testdf):
                logging.info('deleting unwanted Column from train and test')
                df.drop(columns=['ID', 'Experience'], inplace=True, axis=1)
                logging.info('Column removed')
                
                logging.info('Column cleanup')
                df['CCAvg'] = df['CCAvg'] * 12
            
            
            
            
            
            
            
            
            
            
            logging.info('deleting unwanted Column from train and test')
            self.traindf.drop(columns='ID', inplace=True, axis=1)
            self.testdf.drop(columns='ID', inplace=True, axis=1)
            self.traindf = self.traindf.drop(columns='Experience', axis=1)
            self.testdf = self.testdf.drop(columns='Experience', axis=1)
            logging.info('Column removed')
            
            logging.info('Column cleanup')
            self.traindf['CCAvg'] = self.traindf['CCAvg'] * 12
            self.testdf['CCAvg'] = self.testdf['CCAvg'] * 12
            logging.info('Cleanup complete')
            
            logging.info('Removing outlier')
            self.traindf = self.traindf[~(self.traindf['ZIP Code'] < 20000)]
            self.testdf = self.testdf[~(self.testdf['ZIP Code'] < 20000)]
            
            upper_limit_train = self.traindf['Mortgage'].mean() + 3*self.traindf['Mortgage'].std()
            upper_limit_test = self.testdf ['Mortgage'].mean() + 3*self.testdf ['Mortgage'].std()
            
            self.traindf = self.traindf[~(self.traindf['Mortgage'] > upper_limit_train)]
            self.testdf = self.testdf[~(self.testdf['Mortgage'] > upper_limit_test)]
            logging.info('Outlier removed')
            
            logging.info('Splitting input and target variables')
            train_X2 = self.traindf.drop(columns='Personal Loan')
            train_y2 = self.traindf['Personal Loan']
            test_X2 = self.testdf.drop(columns='Personal Loan')
            test_y2 = self.testdf['Personal Loan']
            logging.info('Split successful')
            
            return train_X2, train_y2, test_X2, test_y2
    
        except Exception as e:
            logging.info(f'Error occured in cleaning at: {e}')
            raise CustomException(e, sys)
        
    def transform(self):
        try:
            logging.info('Resampling the data')
            smote = SMOTEN(sampling_strategy='auto', random_state=42)
            resampled_Xtrain2, resampled_Ytrain2 = smote.fit_resample(self.train_X2, self.train_y2)
            logging.info('Resampling done')
            
            logging.info('Scaling the data')
            scaler = StandardScaler()
            scaler.set_output(transform='pandas')
            scaled_Xtrain2 = scaler.fit_transform(resampled_Xtrain2)
            scaled_Xtest2 = scaler.fit_transform(self.test_X2)
            logging.info('Scaling done')