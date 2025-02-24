import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def extract_temporal_features(df):
    df['crawl_year'] = df['crawled_at'].dt.year
    df['crawl_month'] = df['crawled_at'].dt.month
    df['crawl_day'] = df['crawled_at'].dt.day
    df['crawl_weekday'] = df['crawled_at'].dt.weekday
    return df

def encode_categorical(df):
    # Label encoding for brand
    le = LabelEncoder()
    df['brand_encoded'] = le.fit_transform(df['brand'])
    
    # One-hot encoding for category
    ohe = OneHotEncoder(sparse=False)
    category_encoded = ohe.fit_transform(df[['category']])
    df = pd.concat([df, pd.DataFrame(category_encoded, columns=ohe.get_feature_names_out())], axis=1)
    
    return df

def engineer_features(df):
    df = extract_temporal_features(df)
    df = encode_categorical(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv('processed_data.csv')
    df = engineer_features(df)
    df.to_csv('engineered_data.csv', index=False)
