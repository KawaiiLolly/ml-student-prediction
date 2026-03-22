'''
    The file data_ingestion.py is typically used to centralize and automate 
    the process of extracting data from various sources (like databases, APIs, CSV/JSON files, etc.) 
    into a system for further processing, such as transformation and loading into a data warehouse or analytics platform. 

    It serves as a modular component in a data pipeline, ensuring consistent, 
    reusable, and maintainable code for data extraction.
'''
import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import *
from src.utils import save_object

@dataclass
class DataFormationConfig:
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessor_obj_file_path=os.path.join(BASE_DIR,'artifacts', 'preprocessor.pkl') 
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataFormationConfig()

    def get_data_transformer_obj(self):
        '''
            This function is responsible for data transformation
        '''
        
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            
            # creating a pipeline
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()),
                ]
            )
            
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columsn: {numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),  
                    ("categoriacal_pipeline", categorical_pipeline, categorical_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj()
            target_column_name='math score'
            numerical_columns = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on the training and testing dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            # INTERVIEW QUESTION - difference bettween fit_transform() and transform()
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            # INTERVIEW QUESTION - what is the use of np.c_()
            
            logging.info(f"Save preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        