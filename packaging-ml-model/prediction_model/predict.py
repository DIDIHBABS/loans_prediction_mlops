import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_data

classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_prediction(input_data):
    data = pd.DataFrame(input_data)
    prediction = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(prediction == 1,'Y', 'N')
    result = {'Predictions':output}
    return result

# def generate_predictions():
#     test_data = load_data(config.TEST_FILES)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1,'Y','N')
#     print(output)
#     #result = {"Predictions":output}
#     return output

if __name__ == '__main__':
    generate_prediction()