import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


def predict(features):
    try:
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        data_transformed = preprocessor.transform(features)

        pred = model.predict(data_transformed)

        return pred

    except Exception as e:
        logging.info("Exception occurred in prediction")
        raise CustomException(e, sys)