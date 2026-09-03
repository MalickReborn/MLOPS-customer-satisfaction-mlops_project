import logging
import pandas as pd
from zenml import step
from src.evaluate import MSE, R2, RMSE
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin


@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2 score"], Annotated[float, "rmse score"]]:
    """
    Evaluates the odel on the ingested data.

    Args:
        model: the trained model
        X-test: test data
        y_test: label data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculates_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculates_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculates_scores(y_test, prediction)

        return r2, rmse
    except Exception as e:
        logging.error(f"error in evaluating model {e}")
        raise e