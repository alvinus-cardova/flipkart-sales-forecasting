import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, SAVE_DIR

def load_data():
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def clean_prices(df):
    df['actual_price'] = df['actual_price'].str.replace('[^0-9]', '', regex=True).astype(float)
    df['selling_price'] = df['selling_price'].str.replace('[^0-9]', '', regex=True).astype(float)
    df['discount_percentage'] = df['discount'].str.extract('(\d+)').astype(float) / 100
    return df

def process_datetime(df):
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], format='%d/%m/%Y, %H:%M:%S')
    df['year'] = df['crawled_at'].dt.year
    df['month'] = df['crawled_at'].dt.month
    df['day_of_week'] = df['crawled_at'].dt.dayofweek
    return df

def flatten_product_details(df):
    product_details = df['product_details'].apply(
        lambda x: {k: v for d in x for k, v in d.items()}
    )
    return pd.concat([df.drop('product_details', axis=1), pd.json_normalize(product_details)], axis=1)

def handle_missing_values(df):
    num_cols = ['actual_price', 'selling_price', 'discount_percentage', 'average_rating']
    cat_cols = ['brand', 'category', 'sub_category', 'Pattern', 'Color']
    
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in ['brand', 'category', 'sub_category', 'Pattern', 'Color']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess():
    df = load_data()
    df = clean_prices(df)
    df = process_datetime(df)
    df = flatten_product_details(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_csv(f'{SAVE_DIR}/processed.csv', index=False)
    return df
