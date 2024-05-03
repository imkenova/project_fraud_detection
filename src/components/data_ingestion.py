import sys
import os
from src.logger import logging
from src.utils import read_yaml
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass(frozen=True)
class DataIngestionconfig:
   train_data_path:str = os.path.join('artifacts' , 'train.csv')
   test_data_path:str = os.path.join('artifacts' , 'test.csv')
   raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
   def __init__(self) -> None:
      self.ingestion_config = DataIngestionconfig()
      self.schema = read_yaml('SCHEMA.yaml')

   def initiate_data_ingestion(self):
      logging.info('Data Ingestion method starts')
      target = self.schema['target']
      try:
         df = pd.read_csv(os.path.join('notebooks/data','Credit_Card.csv'))
         logging.info("Набор данных прочитан как pandas Dataframe")
         os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False)

         logging.info('Создаются исходные данные')
         target = target['name']
         X = df.drop(target, axis=1)
         y = df[target]

         
         from imblearn.combine import SMOTETomek
         resampler = SMOTETomek(random_state=42)
         X , y = resampler.fit_resample(X, y)
         df=pd.concat([X, y],axis=1)        
         logging.info('Данные разделены на обущающуюи тестовую выборки')
         train_set , test_set = train_test_split(df,test_size=0.3 , random_state=42)
         train_set.to_csv(self.ingestion_config.train_data_path,index =False,header = True)
         test_set.to_csv(self.ingestion_config.test_data_path,index =False,header = True)
         logging.info('Сбор данных завершен')
         return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
         )
      except Exception as e:
         logging.info('Исключение возникло на этапе получения данных')
         raise CustomException(e,sys)
      
if __name__ == "__main__":   
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    print(train_data_path , test_data_path)