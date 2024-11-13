import mlflow
import os
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv('MLFLOW_TRACKING_URI')


def setup_mlflow_autolog(tracking_uri=tracking_uri, experiment_name="unknown"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    mlflow.sklearn.autolog(log_datasets=False)
