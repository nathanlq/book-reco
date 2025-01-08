import pandas as pd
import mlflow
import mlflow.sklearn
from common.setup_mlflow_autolog import setup_mlflow_autolog
from datetime import datetime

setup_mlflow_autolog(experiment_name="compress_prepare_load")

start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with mlflow.start_run(run_name="compress_run"):
    df = pd.read_json('data/raw_output.json')
    df.to_parquet('data/raw_data.parquet')
    mlflow.log_param("source_file", 'data/raw_output.json')
    mlflow.log_param("destination_file", 'data/raw_data.parquet')
    mlflow.log_metric("num_rows", df.shape[0])
    mlflow.log_artifact('data/raw_data.parquet')

    mlflow.log_param("start_time", start_time)

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    mlflow.log_param("end_time", end_time)
