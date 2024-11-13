import pandas as pd
from tqdm import tqdm
import unidecode
import re
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from common.setup_mlflow_autolog import setup_mlflow_autolog
from datetime import datetime

setup_mlflow_autolog()

tqdm.pandas(desc="Processing")

def prepare_data(df):
    with open('data/stop_words_french.txt', 'r', encoding='utf-8') as file:
        french_stop_words = [line.strip() for line in file]

    exploded_labels = df['labels'].explode()
    blacklisted_labels = ['Accueil', 'Collège',
                          'Lycée', 'Livres', 'Littérature']

    def normalize_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = unidecode.unidecode(text)
            text = re.sub(r'\s+', '_', text).strip('_')
        return text

    def clean_label(label):
        label = label.strip()
        if label in blacklisted_labels:
            return None
        if not label:
            return None
        if re.search(r'[0-9]', label) or re.search(r'[\n\t]', label):
            return None
        return normalize_text(label)

    cleaned_labels = exploded_labels.apply(clean_label).dropna()

    cleaned_labels_grouped = cleaned_labels.groupby(level=0).agg(list)

    df['labels'] = cleaned_labels_grouped

    df_info = pd.json_normalize(df['information'])

    df = pd.concat([df.drop(columns=['information']), df_info], axis=1)

    df['Date de parution'] = pd.to_datetime(
        df['Date de parution'], format='%d/%m/%Y')

    df['Nb. de pages'] = df['Nb. de pages'].str.extract(
        r'(\d+)').astype(float).fillna(-1).astype(int)

    df['Poids'] = df['Poids'].str.extract(r'([\d.]+)').astype(float)

    df['EAN'] = df['EAN'].astype(int)

    categorical_columns = ['author', 'Collection',
                           'Editeur', 'Format', 'Présentation']

    for column in categorical_columns:
        df[column] = df[column].apply(normalize_text)

    for column in categorical_columns:
        df[column] = df[column].astype('category')

    for column in categorical_columns:
        df[column + '_label'] = df[column].cat.codes

    df.columns = [normalize_text(col) for col in df.columns]

    dimensions_pattern = r'(\d+,\d+) cm × (\d+,\d+) cm × (\d+,\d+) cm'
    df[['width', 'height', 'depth']] = df['dimensions'].str.extract(
        dimensions_pattern)

    df['width'] = df['width'].str.replace(',', '.').astype(float)
    df['height'] = df['height'].str.replace(',', '.').astype(float)
    df['depth'] = df['depth'].str.replace(',', '.').astype(float)

    df = df.rename(columns={'nb._de_pages': 'nb_de_pages'})

    df.drop(columns=['dimensions'] +
            [col for col in df.columns if '_label' in col], inplace=True)

    return df

if __name__ == "__main__":
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="prepare_run") as run:
        df = pd.read_parquet('data/raw_data.parquet')

        prepared_df = prepare_data(df)

        mlflow.log_param("start_time", start_time)

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        mlflow.log_param("end_time", end_time)

        temp_df = prepared_df.dropna().sample(100)

        prepared_df.to_parquet('data/cleaned_data.parquet')

        mlflow.log_param("source_file", 'data/raw_data.parquet')
        mlflow.log_param("destination_file", 'data/cleaned_data.parquet')

        mlflow.log_metric("num_rows", prepared_df.shape[0])
        mlflow.log_metric("num_columns", prepared_df.shape[1])

        mlflow.log_artifact('data/cleaned_data.parquet')

        input_schema = Schema([
            ColSpec("string", "product_title"),
            ColSpec("string", "author"),
            ColSpec("string", "resume"),
            ColSpec("string", "labels"),
            ColSpec("string", "image_url"),
            ColSpec("string", "collection"),
            ColSpec("datetime", "date_de_parution"),
            ColSpec("long", "ean"),
            ColSpec("string", "editeur"),
            ColSpec("string", "format"),
            ColSpec("string", "isbn"),
            ColSpec("long", "nb_de_pages"),
            ColSpec("double", "poids"),
            ColSpec("string", "presentation"),
            ColSpec("double", "width"),
            ColSpec("double", "height"),
            ColSpec("double", "depth")
        ])

        output_schema = Schema([
            ColSpec("string", "product_title"),
            ColSpec("string", "author"),
            ColSpec("string", "resume"),
            ColSpec("string", "labels"),
            ColSpec("string", "image_url"),
            ColSpec("string", "collection"),
            ColSpec("datetime", "date_de_parution"),
            ColSpec("long", "ean"),
            ColSpec("string", "editeur"),
            ColSpec("string", "format"),
            ColSpec("string", "isbn"),
            ColSpec("long", "nb_de_pages"),
            ColSpec("double", "poids"),
            ColSpec("string", "presentation"),
            ColSpec("double", "width"),
            ColSpec("double", "height"),
            ColSpec("double", "depth")
        ])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        input_example = temp_df.iloc[0:1]

        mlflow.pyfunc.log_model(
            artifact_path="prepare_model",
            python_model=prepare_data,
            signature=signature,
            input_example=input_example
        )

        print(prepared_df.columns)