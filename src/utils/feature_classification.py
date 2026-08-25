import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException


class FeatureClasification:
    def __init__(self, data):
        self.df = pd.read_csv(data)



    def numerical_features(self)->list:
        '''
        Returns the numerical features present in the dataset.
        '''

        try:
            num_features = []
            columns = self.df.columns

            for column in columns:
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    num_features.append(column)
            
            return num_features
        
        except Exception as e:
            raise CustomException(e, sys)



    def categorical_features(self)->list:
        '''
        Returns the categorical features present in the dataset
        '''

        try:
            cat_features = []
            columns = self.df.columns

            for column in columns:
                if pd.api.types.is_object_dtype(self.df[column]) and self.df[column].nunique() < 10:
                    cat_features.append(column)
            
            return cat_features
        
        except Exception as e:
            raise CustomException(e, sys)



    def object_features(self)->list:
        '''
        Returns the object features present in the dataset
        '''

        try:
            obj_features = []
            columns = self.df.columns

            for column in columns:
                if pd.api.types.is_object_dtype(self.df[column]) and self.df[column].nunique() >=10:
                    obj_features.append(column)

            return obj_features
        
        except Exception as e:
            raise CustomException(e, sys)

    

# Test
if __name__ == "__main__":
    fc = FeatureClasification(data='/Users/adityabanerjee/Library/CloudStorage/GoogleDrive-adityabanerjee171@gmail.com/My Drive/Projects/AI Companion Dependency/notebook/data/ai_companion_dependency_dataset.csv')
    print(f"Numerical Features: {fc.numerical_features()}\n\n")
    print(f"Categorical Features: {fc.categorical_features()}\n\n")
    print(f"Object Features: {fc.object_features()}\n\n")
