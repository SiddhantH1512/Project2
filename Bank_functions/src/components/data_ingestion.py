import pandas as pd
import os
import sys
sys.path.append("/Users/siddhant/CustomerSatisfaction/project1")
from logger import logging
from exceptions import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestConfig:
    raw_data1: str=os.path.join("project1/data/artifacts_module1", "raw_data.csv")
    train_data1: str=os.path.join("project1/data/artifacts_module1", "train.csv")
    test_data1: str=os.path.join("project1/data/artifacts_module1", "test.csv")
    
    raw_data2: str=os.path.join("project1/data/artifacts_module2", "raw_data.csv")
    train_data2: str=os.path.join("project1/data/artifacts_module2", "train.csv")
    test_data2: str=os.path.join("project1/data/artifacts_module2", "test.csv")
    
    raw_data3: str=os.path.join("project1/data/artifacts_module3", "raw_data.csv")
    train_data3: str=os.path.join("project1/data/artifacts_module3", "train.csv")
    test_data3: str=os.path.join("project1/data/artifacts_module3", "test.csv")

class InitialiseIngestion:      
    def __init__(self):
        self.ingest = DataIngestConfig()
        
    def startingest(self):
        try:
            logging.info("Pulling data from source")
            df1 = pd.read_csv('/Users/siddhant/CustomerSatisfaction/project1/data/raw/Fraud.csv')
            df2 = pd.read_csv('/Users/siddhant/CustomerSatisfaction/project1/data/raw/loan.csv')
            df3 = pd.read_csv('/Users/siddhant/CustomerSatisfaction/project1/data/raw/churn.csv')
            logging.info("Data pulled into a dataframe")
            
            for df, raw_path, train_path, test_path in zip(
                [df1, df2, df3],
                [self.ingest.raw_data1, self.ingest.raw_data2, self.ingest.raw_data3],
                [self.ingest.train_data1, self.ingest.train_data2, self.ingest.train_data3],
                [self.ingest.test_data1, self.ingest.test_data2, self.ingest.test_data3]):
                
                logging.info("Dropping duplicates")
                df.drop_duplicates(inplace=True)

                
                logging.info("Export dataframe to csv to specified location")
                df.to_csv(raw_path, index=False)

                logging.info("Split data into train and test")
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

                logging.info("Export train data to csv to specified location")
                train_set.to_csv(train_path, index=False)

                logging.info("Export test data to csv to specified location")
                test_set.to_csv(test_path, index=False)

            logging.info("Data exported")
            
            return (
                self.ingest.train_data1, self.ingest.train_data2, self.ingest.train_data3, 
                self.ingest.test_data1, self.ingest.test_data2, self.ingest.test_data3
            )

        
        except Exception as e:
            logging.info(f"The following error occured {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Running data ingestion")
        init_ingest = InitialiseIngestion()
        init_ingest.startingest()
        logging.info("Data ingested")
    except Exception as e:
        logging.info(f"Following error occured: {e}")
        raise CustomException(e, sys)
    
    
    
    
    
