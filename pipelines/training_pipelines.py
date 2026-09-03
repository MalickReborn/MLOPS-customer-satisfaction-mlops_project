#import sys
#from pathlib import Path
# Add project root directory to sys.path
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    config_obj = ModelNameConfig()  # Instance avec la valeur par défaut
    model = train_model(X_train, y_train, config_obj)
    r2, rmse = evaluate_model(model, X_test, y_test)