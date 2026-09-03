import logging
import pandas as pd
import numpy as np
from typing import Union
from sklearn.linear_model import LinearRegression

from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train = pd.DataFrame, y_train = pd.Series) -> None:
        """
        Trains the model
        
        Args:
            X_train: Training datas
            y_train: Training labels

        Returns:
            None
            """
        pass


class LinearRegressionModel(Model):

    """
    Linear Regresion model
    """

    def train(self, X_train, y_train, **kwargs):
        self.model = LinearRegression()
        """
        Trains the model
        Args:
            X_train: tTraining datas
            y_train: Training labels

        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e



        
