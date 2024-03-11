from logger import logging
from exceptions import CustomException
import sys
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
import os

@dataclass
class DataTransformConfig:
    preprocessor_object = os.path.join("artifacts", "pipeline.pkl")
    
class InitiateDataTransformation:
    def __init__(self, dataframe):
        self.transform_config = DataTransformConfig()
        self.df = dataframe
        self.clean = self.clean()
        
    
    def clean(self):
        try:
            logging.info('Creating new features')
            self.df['org_balance_change'] = self.df.apply(lambda x: 1 if (x['newbalanceOrig'] - x['oldbalanceOrg']) > 0 else 0, axis=1)
            self.df['dest_balance'] = self.df.apply(lambda x: abs(x['newbalanceDest'] - x['oldbalanceDest']), axis=1)
            self.df['dest_balance_change'] = self.df.apply(lambda x: 1 if (x['newbalanceDest'] - x['oldbalanceDest']) > 0 else 0, axis=1)
            logging.info('New features created')
            
            logging.info("Dropping columns")
            self.df.drop(["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 'nameOrig', 'nameDest'],axis=1,inplace=True)
            logging.info("Dropped columns")
            
            logging.info('Splitting input and target variables')
            train_X = self.df.drop(columns='isFraud')
            train_y = self.df['isFraud']
            logging.info('Split successful')
            
            return train_X, train_y
            
        except Exception as e:
            logging.info('Following error occured while cleaning: {e}')
            raise CustomException(e, sys)
    
    def transform(self):
        try:
            encode = ['type']
            scale = ['step', 'amount', 'dest_balance']
            
            logging.info(f'Encoding feature {encode}')
            logging.info(f'Scaling features {scale}')
            transformer = ColumnTransformer(transformers=[
                ('ohe', OneHotEncoder(sparse_output=False), encode),
                ('scale', StandardScaler(), scale)
            ], remainder='passthrough')
            
            
            pipeline = Pipeline([
                ('transform', transformer),
                ('resample', SMOTEN(sampling_strategy='auto', random_state=42))
            ])
            
            X_train_resampled, y_train_resampled = pipeline.fit_resample(self.train_X, self.train_y)
            logging.info('Resampling, and transformations completed')
            
            return X_train_resampled, y_train_resampled
        
        except Exception as e:
            logging.info('Error occured while transformation at: {e}')
            raise CustomException(e, sys)
    
    def initiate_transform(self, traindf, testdf):
        try:
            logging.info('Creating transform object')
            transform_obj = self.transform()
            logging.info('object created')
            
            logging.info('Loading train and test data')
            
            