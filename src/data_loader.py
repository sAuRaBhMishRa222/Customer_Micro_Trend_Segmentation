import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    df['hour'] = df['TransactionDate'].dt.hour
    df['dayofweek'] = df['TransactionDate'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['TransactionDate'].dt.month

    return df
