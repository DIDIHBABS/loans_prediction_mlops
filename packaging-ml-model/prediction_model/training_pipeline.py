import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_data, save_pipeline
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe


def perform_training():
    train_data = load_data(config.TRAIN_FILES)
    train_y = train_data[config.TARGET].map({'N':0,'Y':1})
    pipe.classifications_pipeline.fit(train_data[config.FEATURES],train_y)
    save_pipeline(pipe.classifications_pipeline)

if __name__=='__main__':
    perform_training()