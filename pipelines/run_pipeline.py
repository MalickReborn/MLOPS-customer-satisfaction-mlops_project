import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from pipelines.training_pipelines import train_pipeline


def main():
    #launch = train_pipeline(data_path="/home/malick/MicrosoftMlOps/zenml-projects/customer-satisfaction-mlops_project/data/olist_customers_dataset.csv")
    #deploy = "zenml pipeline deploy pipelines.launch"
    #return deploy
    train_pipeline(data_path="/home/malick/MicrosoftMlOps/zenml-projects/customer-satisfaction-mlops_project/data/olist_customers_dataset.csv")

if __name__ == "__main__":
    main()
