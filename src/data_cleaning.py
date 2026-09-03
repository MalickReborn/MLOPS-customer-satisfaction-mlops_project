import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple
from typing_extensions import Annotated
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        logging.info("Preprocessing data")
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median()),
            data["product_length_cm"].fillna(data["product_length_cm"].median()),
            data["product_height_cm"].fillna(data["product_height_cm"].median()),
            data["product_width_cm"].fillna(data["product_width_cm"].median()),
            data["review_comment_message"].fillna("No review"),
            # Simple pandas median imputation
            # Simple pandas median imputation
            data.fillna(data.median(numeric_only=True))
            # Numerical columns imputation
            data = data.dropna()

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        
        except Exception as e:
            logging.error(f"Error in processing data: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy to Divide data for train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try: 
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


        

class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """

    def __init__(self, data: pd.DataFrame, strategy= DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e

if __name__ == "__main__":
    data = pd.read_csv("/home/malick/MicrosoftMlOps/zenml-projects/customer-satisfaction-mlops_project/data/olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    data_cleaning.handle_data()

