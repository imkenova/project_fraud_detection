import mlflow.sklearn 
from mlflow import log_metric, log_params, log_artifact 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score  
from src.exception import CustomException  
from src.logger import logging 
from src.utils import (save_object, read_yaml) 
from dataclasses import dataclass  
import os, sys
import warnings

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri('https://dagshub.com/imkenova/project_fraud_detection.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'imkenova'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '2a258ef043924179df455e1024a246bfed159099'


@dataclass(frozen=True)
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerconfig()
        self.params: dict = read_yaml('PARAMS.yaml')

    def initiate_model_training(self, train_arr, test_arr):
        params = self.params['params']
        log_params(params)
        try:
            mlflow.sklearn.autolog(silent=True)
            logging.info('Разделение зависимых и независимых переменных из обучающих и тестовых данных')
            X_train , y_train , X_test , y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )


            logging.info("Инициализация Random Forest")
            model = RandomForestClassifier(random_state=2)
            model.set_params(**params) 
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            log_metric('accuracy', score)

            logging.info(f"оценка точности RandomForestClassifier составляет : {score*100:.2f}%")


            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            log_artifact(local_path=self.model_trainer_config.trained_model_file_path)
            mlflow.end_run()
            logging.info("Обучение модели завершено, модель сохранена в под именем model.pkl")
        except Exception as e:
            raise CustomException(e, sys)