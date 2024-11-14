import mlflow
import os
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

mlflow_host = os.getenv('MLFLOW_HOST', 'localhost')
mlflow_port = os.getenv('MLFLOW_PORT', '5000')
tracking_uri = f"http://{mlflow_host}:{mlflow_port}"


def setup_mlflow_autolog(tracking_uri=tracking_uri, experiment_name="unknown"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    mlflow.sklearn.autolog(log_datasets=False)
