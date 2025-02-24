import json
import pandas as pd
from datetime import datetime
import re

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def clean_price(price_str):
    if isinstance(price_str, str):
        return float(re.sub(r'[^\d.]', '', price_str.replace(',', '')))
    return price_str

def parse_date(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y, %H:%M:%S')

def preprocess_data(df):
    # Clean prices
    df['actual_price'] = df['actual_price'].apply(clean_price)
    df['selling_price'] = df['selling_price'].apply(clean_price)
    
    # Extract discount percentage
    df['discount'] = df['discount'].apply(lambda x: float(re.findall(r'\d+', x)[0]) if isinstance(x, str) else 0)
    
    # Parse datetime
    df['crawled_at'] = df['crawled_at'].apply(parse_date)
    
    # Flatten product_details
    for detail in df['product_details']:
        for item in detail:
            key = list(item.keys())[0]
            df[key] = df.get(key, '') + ';' + item[key]
    
    # Drop unnecessary columns
    df.drop(['_id', 'url', 'images', 'product_details'], axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    df = load_data('flipkart_fashion_products_dataset.json')
    df = preprocess_data(df)
    df.to_csv('processed_data.csv', index=False)
