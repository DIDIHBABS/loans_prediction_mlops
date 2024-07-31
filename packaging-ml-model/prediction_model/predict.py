import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_data, load_pipeline, save_pipeline


classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_prediction(input_data):
    data = pd.DataFrame(input_data)
    pred = classification_pipeline(data[config.FEATURES])
    output = np.where(predictions==1,'Y', 'N')
    result = {'Predictions':output}
    return result

if __name__ = '__main__':
    generate_prediction()