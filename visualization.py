import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')

def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['selling_price'], kde=True)
    plt.title('Price Distribution')
    plt.savefig('price_distribution.png')

if __name__ == "__main__":
    df = pd.read_csv('engineered_data.csv')
    plot_price_distribution(df)
    
    # Load model for feature importance
    rf_model = joblib.load('random_forest_model.pkl')
    plot_feature_importance(rf_model, df.drop('selling_price', axis=1).columns)
