import logging

import pandas as pd
from zenml import step

# Creation of a data Ingestion Class
class IngestData:
    
    #We first create the data object
    def __init__(self, data_path: str):
        """
            Args:
                data_path: path to the data
        """
        self.data_path = data_path
    
    #we create the function that turn the data object into an ingested data under a type of panda Dataframe
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data   
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.info(f"Error while ingesting data: {e}")
        raise e