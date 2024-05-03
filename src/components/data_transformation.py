from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os , sys
from dataclasses import dataclass
from src.utils import save_object
from src.utils import read_yaml


@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    STATUS_FILE: str = os.path.join('artifacts','data_validation_status.txt')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.schema = read_yaml('SCHEMA.yaml')

    def get_data_transformation_object(self):

        try:
            logging.info("Data transformation initialized")

        
            cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
            numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']

            logging.info('Data transformation columns created')

            numerical_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OneHotEncoder(drop='first',handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                    ]
                )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline, numerical_cols),
                ('categorical_pipeline',cat_pipeline , cat_cols)
                ])
            logging.info('Data Transformation Preprocessor loaded successfully')

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in Data Transformation Preprocessor")
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        with open(self.data_transformation_config.STATUS_FILE , 'r') as f:
            text = f.read()

        status = text.split(':')[-1].strip()

        if status:
            try:

                train_df = pd.read_csv(train_data_path)
                test_df = pd.read_csv(test_data_path)
                schema = self.schema['target']
                input_feature_df = pd.concat([train_df.assign(ind='train'), test_df.assign(ind='test')])

                logging.info("Read train and test data completed")
                logging.info(f'Train Datatrame Head : \n {train_df.head().to_string()}')
                logging.info(f'Test Datatrame Head : \n {test_df.head().to_string()}')

                preprocessing_obj = self.get_data_transformation_object()

                target_column = schema['name']

             
                target_feature_train_df = input_feature_df[input_feature_df['ind'].eq('train')][target_column]
                target_feature_test_df = input_feature_df[input_feature_df['ind'].eq('test')][target_column]
                input_feature_df = input_feature_df.drop(columns=target_column,axis=1)

                

                input_feature_df['SEX'] = input_feature_df['SEX'] - 1
                input_feature_df['EDUCATION'] = input_feature_df['EDUCATION'].apply(lambda x: 4 if x >= 4 else x)
                input_feature_df['PAY_0'] = input_feature_df['PAY_0'].apply(lambda x: 0 if x<=0 else x)
                input_feature_df['PAY_2'] = input_feature_df['PAY_2'].apply(lambda x: 0 if x<=0 else x)
                input_feature_df['PAY_3'] = input_feature_df['PAY_3'].apply(lambda x: 0 if x<=0 else x)
                input_feature_df['PAY_4'] = input_feature_df['PAY_4'].apply(lambda x: 0 if x<=0 else x)
                input_feature_df['PAY_5'] = input_feature_df['PAY_5'].apply(lambda x: 0 if x<=0 else x)
                input_feature_df['PAY_6'] = input_feature_df['PAY_6'].apply(lambda x: 0 if x<=0 else x)


                
                input_feature_train_df = input_feature_df[input_feature_df['ind'].eq('train')]
                input_feature_test_df = input_feature_df[input_feature_df['ind'].eq('test')]

                
                input_feature_train_df=input_feature_train_df.drop(['ind'],axis=1)
                input_feature_test_df=input_feature_test_df.drop(['ind'],axis=1)

                
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                logging.info("Applying preprocessing objects on training data and test data")

                train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessing_obj
                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )

            except Exception as e:
                logging.info("Error occured while applying preprocessing objects")
                raise CustomException(e,sys)

        else:
            logging.info('Error occured in data validation kindly check the data')
            raise ValueError('Error in data validation')