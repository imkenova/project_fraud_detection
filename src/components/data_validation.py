import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml
import pandas as pd
from dataclasses import dataclass , field

@dataclass(frozen=True)
class DataValidationconfig:
   data_path: str = os.path.join('artifacts','raw.csv')
   STATUS_FILE: str = os.path.join('artifacts','data_validation_status.txt')
   ALL_REQUIRED_FILES: list[str] = field(default_factory=lambda:['raw.csv','train.csv','test.csv'])

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationconfig()
        self.schema = read_yaml('SCHEMA.yaml')

    def initiate_data_validation(self):
        logging.info('Data validation started')
        try:
            schema = self.schema['columns']
            df = pd.read_csv(self.validation_config.data_path)
            flag = True
            all_files = os.listdir('artifacts')
            for file in self.validation_config.ALL_REQUIRED_FILES:
                if file not in all_files or dict(df.dtypes) != schema :
                    flag = False
                    break
            validation_status = flag
            with open(self.validation_config.STATUS_FILE,'w') as f:
                f.write(f"Validation status: {validation_status}")
            logging.info(f"Validation status: {validation_status}")
        except Exception as e:
            logging.info("Error occurred while data validation")
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataValidation()
    obj.initiate_data_validation()