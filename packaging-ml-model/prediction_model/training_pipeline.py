import pandas as pd
import numpy as np
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeliine as pipe
from prediction_model.processing.data_handling import load_data, load_pipeline, save_pipeline
import sys
def perform_training():
    train_data = load_data(config.TRAIN_FILES)
    train_y = train_data[config.TARGET_FEATURE].map({'N':0, 'Y':1})
    train_x = train_data[config.FEATURES]
    pipe.classifications_pipeline.fit(train_x, train_y)
    save_pipeline(pipe.classifications_pipeline)

    if __name__ =='__main__':
        perform_training()
