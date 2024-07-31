import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from prediction_model.config import config

class DemoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,):
        pass
    def fit(self, X,y=None):
        return self
    def transform(self):
        return X


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mean_dict = {}
    def fit(self, X, y=None ):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        X =X.copy()
        for  col in self.variables:
            X[col].fillna(self.mean_dict[col], inplace=True)
            return X


class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mode_dict = {}

    def fit(self, X, y):
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()
        return self

    def transform(self, X):
        for col in self.variables:
            X[col].fillna(self.mode_dict[col], inplace=True)


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns= self.variables_to_drop)
        return X


class FeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_add=None, variables_to_modify=None):
        self.variables_to_add = variables_to_add
        self.variables_to_modify = variables_to_modify

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in  self.variables_to_modify:
                X[feature] = X[feature] + X[self.variables_to_add]
        return X

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.label_dict = {}

    def fit(self, X, y):
        for col in self.variables:
            t = X[col].values_counts().sort_values(ascending=True).index
            self.label_dict[col] = {k:i for i,k in enumerate(t,0)}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col]= X[col].map(self.label_dict[col])
        return X


# lOG TRANSFORMER
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables


    def fit(self, X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X