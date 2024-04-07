import os, sys
import pickle
import yaml
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score , roc_auc_score , f1_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj:
            pickle.dump(obj , file_obj)

    except Exception as e:
        logging.info('Exception occured during save_object utils')
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report ={}
        for i in range(len(models)):
            model = list(models.values())[i]
            #Train model
            model.fit(X_train , y_train)

            #Predict testing data
            y_test_pred = model.predict(X_test)

            #Get the accuracy scores for train and test data

            #Train_model_score = accuracy_score(y_train , y_train_pred)
            test_model_score = accuracy_score(y_true=y_test , y_pred=y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')

        def read_yaml(path_to_yaml) -> dict:
            """reads yaml file and returns ConfigBox
            Args:
                path_to_yaml (str): path like input

            Raises:
                ValueError: if yaml file is empty
                e: empty yaml file
            Returns:
                dict: dictionary of the yaml file contents
            """
            try:
                with open(path_to_yaml) as yaml_file:
                    content = yaml.safe_load(yaml_file)
                    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
                    return content
            except Exception as e:
                raise e