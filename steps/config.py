from pydantic import BaseModel
from zenml import step

class ModelNameConfig(BaseModel):
    #model configs
    model_name : str = "LinearRegression"
