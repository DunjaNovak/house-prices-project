import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Popuni nedostajuće vrednosti
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("None")

    # One-hot encoding kategorija
    df = pd.get_dummies(df, drop_first=True)
    return df


def advanced_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Popuna vrednosti
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    for col in num_cols:
        if df[col].skew() > 0.75:
            df[col] = np.log1p(df[col])  # log transformacija
        df[col] = df[col].fillna(df[col].median())
    
    for col in cat_cols:
        df[col] = df[col].fillna("None")
    
    df = pd.get_dummies(df, drop_first=True)
    return df


def scale_data(train_df, test_df, method="robust"):
    if method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled
