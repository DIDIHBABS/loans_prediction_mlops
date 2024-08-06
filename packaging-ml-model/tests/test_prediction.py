from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_data
from prediction_model.predict import generate_prediction

@pytest.fixture
def single_prediction():
    test_data = load_data(config.TEST_FILES)
    single_row = test_data[:1]
    result = generate_prediction(single_row)
    return result


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None


def test_single_pred_type(single_prediction):
    assert isinstance(single_prediction.get('Predictions')[0], str)

def test_single_is_(single_prediction):
    assert single_prediction.get('Predictions')[0] =='Y'