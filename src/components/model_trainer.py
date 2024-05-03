from mlflow import log_metric , log_params , log_artifact
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import  accuracy_score 
from src.exception import CustomException  
from src.logger import logging 
from src.utils import (save_object , read_yaml) 
from dataclasses import dataclass 
import os , sys


@dataclass(frozen=True)
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerconfig()
        self.params: dict = read_yaml('PARAMS.yaml')

    def initiate_model_training(self,train_arr,test_arr):
        params = self.params['params']
        log_params(params)
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )


            logging.info("Random Forest initialized")
            model = RandomForestClassifier(random_state=2)
            model.set_params(**params) 
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test,y_pred)
            log_metric('accuracy', score)

            logging.info(f"accuracy score of RandomForestClassifier is : {score*100:.2f}%")


            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )
            log_artifact(local_path=self.model_trainer_config.trained_model_file_path)
            logging.info("Model training complete and model saved in artifacts as model.pkl")
        except Exception as e:
            raise CustomException(e,sys)