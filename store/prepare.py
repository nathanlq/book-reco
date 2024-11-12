import pandas as pd
from tqdm import tqdm
import unidecode
import re

tqdm.pandas(desc="Processing")

df = pd.read_parquet('data/raw_data.parquet')

with open('data/stop_words_french.txt', 'r', encoding='utf-8') as file:
    french_stop_words = [line.strip() for line in file]

exploded_labels = df['labels'].explode()
blacklisted_labels = ['Accueil', 'Collège', 'Lycée', 'Livres', 'Littérature']


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

df.drop(columns=['dimensions'] + [col for col in df.columns if '_label' in col], inplace=True)

df.to_parquet('data/cleaned_data.parquet')

print(df.columns)