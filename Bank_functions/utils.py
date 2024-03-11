import numpy as np 
import pandas as pd
from exceptions import CustomException
import sys
import pickle 

def save_obj(file_path, obj):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)