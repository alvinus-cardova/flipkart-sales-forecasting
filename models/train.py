from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import joblib

def load_processed_data():
    return pd.read_csv(
        'processed_data/processed.csv',
        dtype={'discount_percentage': float},
        low_memory=False
    )

from sklearn.ensemble import StackingRegressor

def novel_approach():
    df = load_processed_data()
    
    # Explicitly select only numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_cols].drop(['selling_price'], axis=1)
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    base_models = [
        ('dt', DecisionTreeRegressor()),
        ('rf', RandomForestRegressor(n_estimators=50))
    ]
    
    stack = StackingRegressor(estimators=base_models, final_estimator=GradientBoostingRegressor())
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)
    
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }

def train_models():
    df = load_processed_data()
    X = df.drop(['selling_price', 'crawled_at', '_id', 'description', 'url', 'pid'], axis=1)
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {'MAE': mae, 'MSE': mse}
        joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')
    
    return results
