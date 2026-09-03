import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):

    """
    abstract class defining strategy for evaluation of our models
    """


    @abstractmethod
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True Labels
            y_pred Predicted Labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squarres error
    """
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse

        except Exception as e:
            logging.info(f"Error in calculating R2 Score MSE evaluation: {e}")
            raise e

class R2(Evaluation):

    """
    Evaluation Strategy that uses R2 Score
    """

    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 score: {r2}")
            return r2
        
        except Exception as e:
            logging.info(f"Error in calculating R2 Score: {e}")
            raise e

class RMSE(Evaluation):

    """
        Evaluation Strategy that uses Root Mean Squared Error
    """
    
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            rmse = root_mean_squared_error(y_true, y_pred,)
            logging.info(f"RMSE: {rmse}")
            return rmse
        
        except Exception as e:
            logging.info(f"Error in calculating RMSE Score: {e}")
            raise e