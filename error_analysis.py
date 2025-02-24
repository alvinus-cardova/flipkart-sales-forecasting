import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_error_table(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }

if __name__ == "__main__":
    # Example usage after model prediction
    df = pd.read_csv('engineered_data.csv')
    X_test = df.drop('selling_price', axis=1)
    y_test = df['selling_price']
    
    rf_model = joblib.load('random_forest_model.pkl')
    predictions = rf_model.predict(X_test)
    
    error_table = generate_error_table(y_test, predictions)
    print("Error Analysis Table:", error_table)
