import mlflow
import os
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

mlflow_host = os.getenv('MLFLOW_HOST', '127.0.0.1/')
mlflow_port = os.getenv('MLFLOW_PORT', '5000')
mlflow_login = os.getenv('MLFLOW_LOGIN')
mlflow_password = os.getenv('MLFLOW_PASSWORD')
tracking_uri = f"https://{mlflow_host}"

def setup_mlflow_autolog(tracking_uri=tracking_uri, experiment_name="unknown"):
    """
    Configure MLflow pour un serveur distant avec authentification.

    Args:
        tracking_uri (str): URI de suivi du serveur MLflow.
        experiment_name (str): Nom de l'expérience MLflow.
    """
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(log_datasets=False)
    print(f"MLflow configuré pour suivre l'URI : {tracking_uri}")
    print(f"Expérience active : {experiment_name}")
