from preprocessing.preprocess import preprocess
from features.engineering import create_features
from models.train import train_models
from visualization.visualize import plot_metrics, plot_feature_importance
import joblib

def main():
    # Preprocess data
    df = preprocess()
    
    # Feature engineering
    df = create_features(df)
    
    # Train models
    results = train_models()
    
    # Visualize results
    plot_metrics(results)
    
    # Load best model for feature importance
    best_model = joblib.load('models/Random_Forest.pkl')
    df_processed = pd.read_csv('processed_data/processed.csv')
    feature_names = df_processed.drop(['selling_price', 'crawled_at', '_id', 'description', 'url', 'pid'], axis=1).columns
    plot_feature_importance(best_model, feature_names)

if __name__ == '__main__':
    main()
