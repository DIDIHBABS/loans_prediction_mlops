import os
import joblib
import pandas as pd
import prediction_model.config import config


def load_data(file_name):
    file_path = os.path.join(config.DATASET,file_name)

def save_pipeline(pipe_line_to_save):
    save_path = os.path.join(config.MODEL_NAME)
    joblib.dump(pipe_line_to_save, save_path)
    print(f"Model had been stored to{config.MODEL_NAME}")

def load_pipeline(pipe_line_to_load):
    save_path = os.path.join(config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print('Model has been loaded')
    return model_loaded