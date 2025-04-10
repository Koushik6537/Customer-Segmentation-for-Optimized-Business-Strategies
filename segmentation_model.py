import pandas as pd
from sklearn.cluster import KMeans

def segment_customers(df):
    required_columns = ['age', 'price']

    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Drop customer_id if present
    if "customer_id" in df.columns:
        df.drop("customer_id", axis=1, inplace=True)

    # Encode gender if present
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    X = df[['age', 'price']].values

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    return df, kmeans.cluster_centers_
