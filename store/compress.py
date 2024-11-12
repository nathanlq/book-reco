import pandas as pd

df = pd.read_json('data/raw_output.json')
df.to_parquet('data/raw_data.parquet')
