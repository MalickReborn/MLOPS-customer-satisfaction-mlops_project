import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.clean_data import clean_df
from steps.config import ModelNameConfig

@step 
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:

    """
    Trains the model on the ingested data
    
    Args:
        X_train : pd.Dataframe
        X_test : pd.DataFrame
        y_train : pd.DataSeries
        y_test : pd.DataSeries
        config = ModelNameConfig

"""
    model = None
    if config.model_name == "LinearRegression":
        model = LinearRegressionModel()
        model = model.train(X_train, y_train)
        return model   
    else:
        raise ValueError(f"Model {config.model_name} not supported") 