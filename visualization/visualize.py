import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metrics(results):
    df = pd.DataFrame(results).T
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization/model_comparison.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('visualization/feature_importance.png')
    plt.close()
