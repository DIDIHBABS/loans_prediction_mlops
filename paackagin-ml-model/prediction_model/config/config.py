import os
import pathlib
import prediction_model

#files and file paths

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATASET = os.path.join(PACKAGE_ROOT, 'datasets')
TRAIN_FILES = 'train.csv'
TEST_FILES = 'test.csv'

MODEL_NAME = 'classification.pkl'
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

# Features
TARGET_FEATURE = 'Loan_Status'
DROP_FEATURES = ['CoapplicantIncome']
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
NUMERICAL_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CATEGORICAL_FEATURES = ['Gender','Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area'],['ApplicantIncome'], ['CoapplicantIncome']

LOG_TRANSFORM = ['ApplicantIncome', 'LoanAmount']
SCALED_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area']
FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'