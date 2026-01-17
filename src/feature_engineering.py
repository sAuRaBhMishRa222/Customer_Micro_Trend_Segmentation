from sklearn.preprocessing import StandardScaler

def create_customer_features(df):
    customer_features = df.groupby('CustomerID').agg(
        total_orders=('TransactionDate', 'count'),
        avg_quantity=('Quantity', 'mean'),
        avg_price=('Price', 'mean'),
        avg_discount=('DiscountApplied(%)', 'mean'),
        avg_order_value=('TotalAmount', 'mean'),
        night_ratio=('hour', lambda x: (x >= 22).mean()),
        weekend_ratio=('is_weekend', 'mean'),
        category_diversity=('ProductCategory', 'nunique')
    ).reset_index()

    X = customer_features.drop('CustomerID', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return customer_features, X_scaled
