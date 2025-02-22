import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, SAVE_DIR

def load_data():
    chunks = []
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
        # Process in chunks of 100,000 records
        for i in range(0, len(data), 100000):
            chunks.append(pd.json_normalize(data[i:i+100000]))
    return pd.concat(chunks, ignore_index=True)

def clean_prices(df):
    for col in ['actual_price', 'selling_price']:
        # Use .loc to avoid chained assignment
        df.loc[:, col] = (
            df[col]
            .str.replace('[^0-9.]', '', regex=True)
            .replace('', pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(df[col].median())
        )
    
    # Alternative for discount percentage
    df = df.assign(
        discount_percentage=df['discount'].str.extract(r'(\d+)', expand=False)
                                           .astype(float)
                                           .div(100)
                                           .fillna(0)
    )
    return df

def process_datetime(df):
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], format='%d/%m/%Y, %H:%M:%S')
    df['year'] = df['crawled_at'].dt.year
    df['month'] = df['crawled_at'].dt.month
    df['day_of_week'] = df['crawled_at'].dt.dayofweek
    return df

def flatten_product_details(df):
    # Check if column exists
    if 'product_details' not in df.columns:
        raise ValueError("product_details column missing - check data loading")
    
    # Explode and normalize
    df_exploded = df.explode('product_details').reset_index(drop=True)
    details_df = pd.json_normalize(df_exploded['product_details'])
    return pd.concat([df_exploded.drop('product_details', axis=1), details_df], axis=1)

def handle_missing_values(df):
    # Numerical columns
    num_cols = ['average_rating']
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
    
    # Categorical columns
    cat_cols = ['brand', 'category', 'sub_category', 'Pattern', 'Color']
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in ['brand', 'category', 'sub_category', 'Pattern', 'Color']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess():
    # 1. Load raw data
    df = load_data()
    
    # 2. Clean numerical fields
    df = clean_prices(df)
    
    # 3. Process datetime
    df = process_datetime(df)
    
    # 4. Flatten nested structures
    df = flatten_product_details(df)
    
    # 5. Handle missing values
    df = handle_missing_values(df)
    
    # 6. Encode categoricals
    df = encode_categorical(df)
    
    # Save processed data
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_csv(f'{SAVE_DIR}/processed.csv', index=False)
    return df
