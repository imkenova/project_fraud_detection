import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

STAGE_NAME = "Сбор данных"
try:
    logging.info(f">>>>>>> этап {STAGE_NAME} начат <<<<<<<<<<")
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    logging.info(f">>>>>>>> этап {STAGE_NAME} завершен <<<<<<<<<<\n\nx================x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e,sys)


STAGE_NAME = "Проверка данных"
try:
    logging.info(f">>>>>>> этап {STAGE_NAME} начат <<<<<<<<<<")
    data_validation = DataValidation()
    data_validation.initiate_data_validation()
    logging.info(f">>>>>>>> этап {STAGE_NAME} завершен <<<<<<<<<<\n\nx================x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e,sys)


STAGE_NAME = "Преобразование данных"
try:
    logging.info(f">>>>>>> этап {STAGE_NAME} начат <<<<<<<<<<")
    data_transformation = DataTransformation()
    train_arr, test_arr , obj_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    logging.info(f">>>>>>>> этап {STAGE_NAME} завершен <<<<<<<<<<\n\nx================x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e,sys)


STAGE_NAME = "Обучение модели"
try:
    logging.info(f">>>>>>> этап {STAGE_NAME} начат <<<<<<<<<<")
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
    logging.info(f">>>>>>>> этап {STAGE_NAME} завершен <<<<<<<<<<\n\nx================x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e,sys)