from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd

def balance_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    return X_res, y_res, X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv('engineered_data.csv')
    # Assuming 'out_of_stock' is the target variable for classification
    X, y, X_test, y_test = balance_data(df, 'out_of_stock')
