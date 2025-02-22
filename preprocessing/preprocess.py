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
    # Process prices and discounts first
    df = df.copy()
    
    # Remove original discount column after extracting percentage
    df = df.drop('discount', axis=1)
    # Clean and convert price columns
    for col in ['actual_price', 'selling_price']:
        # Step 1: Remove non-numeric characters
        cleaned = df[col].str.replace(r'[^0-9.]', '', regex=True)
        
        # Step 2: Convert to numeric type
        numeric_vals = pd.to_numeric(cleaned, errors='coerce')
        
        # Step 3: Calculate median from cleaned numeric values
        median_val = numeric_vals.median()
        
        # Step 4: Fill missing values and update dataframe
        df.loc[:, col] = numeric_vals.fillna(median_val)
    
    # Clean discount percentage
    df['discount_percentage'] = (
        df['discount']
        .str.extract(r'(\d+)', expand=False)
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
    # After exploding and normalizing product_details
    df = pd.concat([...], axis=1)
    
    # Clean any percentage columns from product_details
    percentage_cols = [col for col in df.columns if '%' in col]
    for col in percentage_cols:
        df[col] = df[col].str.replace('%', '').astype(float)
    
    return df

def handle_missing_values(df):
    # Convert average_rating to numeric first
    df['average_rating'] = pd.to_numeric(
        df['average_rating'].str.strip().replace('', pd.NA),
        errors='coerce'
    )
    
    # Fill numeric columns
    num_cols = ['average_rating']
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Handle categorical columns
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
