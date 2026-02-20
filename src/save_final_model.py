import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from src.feature_engineering import remove_outliers, advanced_feature_engineering
from src.utils import advanced_preprocess_data

def main():
    print("📌 Učitavanje train podataka...")
    df = pd.read_csv("data/train.csv")
    df = remove_outliers(df)
    df = advanced_feature_engineering(df)

    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])
    X = advanced_preprocess_data(X).fillna(0)

    print("📌 Trening finalnog modela (Ridge)...")
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "models/final_model.pkl")
    print("✅ Sačuvan final_model.pkl u folder models/")

if __name__ == "__main__":
    main()
