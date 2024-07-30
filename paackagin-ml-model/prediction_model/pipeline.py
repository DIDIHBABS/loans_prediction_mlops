from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

classificationS_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables=config.NUMERICAL_FEATURES)),
        ('ModeImputation', pp.MeanImputer(variables=config.CATEGORICAL_FEATURES)),
        ("DropFeatures", pp.DropColumns(variabes=config.DROP_FEATURES )),
        ('FeatureTransform', pp.FeatureTransform(variable_to_modify = config.FEATURE_TO_MODIFY,
        variable_to_add = config.FEATURE_TO_ADD)),
        ('LabelEncoder', pp.LabelEncoder(variables=config.CATEGORICAL_FEATURES)),
        ('LogTransformer', pp.LogTransformer(variables=config.LOG_TRANSFORM)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))]
)