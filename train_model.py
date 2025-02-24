from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib

def train_evaluate_model(X_train, y_train, X_test, y_test, model_type='decision_tree'):
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boost':
        model = GradientBoostingRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, {'MAE': mae, 'MSE': mse, 'R2': r2}

if __name__ == "__main__":
    df = pd.read_csv('engineered_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop('selling_price', axis=1), df['selling_price'], test_size=0.2)
    
    # Train models
    dt_model, dt_metrics = train_evaluate_model(X_train, y_train, X_test, y_test, 'decision_tree')
    rf_model, rf_metrics = train_evaluate_model(X_train, y_train, X_test, y_test, 'random_forest')
    
    # Save models
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    
    print("Decision Tree Metrics:", dt_metrics)
    print("Random Forest Metrics:", rf_metrics)
