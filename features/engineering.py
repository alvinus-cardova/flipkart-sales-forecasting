import pandas as pd

def create_features(df):
    df['price_diff'] = df['actual_price'] - df['selling_price']
    df['is_discounted'] = (df['discount_percentage'] > 0).astype(int)
    return df
