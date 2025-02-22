import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, SAVE_DIR

def load_data():
    with open(DATA_PATH, 'r') as f:
        # Load entire JSON array
        data = json.load(f)
    
    # Normalize nested structures
    df = pd.json_normalize(
        data,
        meta=[
            "_id", "actual_price", "average_rating", "brand", "category",
            "crawled_at", "description", "discount", "out_of_stock", "pid",
            "seller", "selling_price", "sub_category", "title", "url"
        ],
        record_path="product_details",
        errors="ignore"
    )
    return df

def clean_prices(df):
    # Clean price columns
    for col in ['actual_price', 'selling_price']:
        # Remove non-numeric characters (preserve decimals)
        df[col] = df[col].str.replace('[^0-9.]', '', regex=True)
        
        # Replace empty strings with NaN
        df[col] = df[col].replace('', pd.NA)
        
        # Convert to numeric type and fill missing values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    # Clean discount percentage
    df['discount_percentage'] = (
        df['discount']
        .str.extract('(\d+)', expand=False)
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
    # Explode the product_details list
    df_exploded = df.explode("product_details").reset_index(drop=True)
    
    # Extract keys from dictionaries in product_details
    df_details = pd.json_normalize(df_exploded["product_details"])
    
    # Combine with original data
    return pd.concat([df_exploded.drop("product_details", axis=1), df_details], axis=1)

def handle_missing_values(df):
    # Handle remaining missing values
    num_cols = ['average_rating']
    cat_cols = ['brand', 'category', 'sub_category', 'Pattern', 'Color']
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
        
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in ['brand', 'category', 'sub_category', 'Pattern', 'Color']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess():
    # Load and normalize data
    df = load_data()
    
    # Clean prices and discounts
    df = clean_prices(df)
    
    # Process datetime
    df = process_datetime(df)
    
    # Flatten nested product_details
    df = flatten_product_details(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categoricals
    df = encode_categorical(df)
    
    # Save processed data
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_csv(f'{SAVE_DIR}/processed.csv', index=False)
    return df
