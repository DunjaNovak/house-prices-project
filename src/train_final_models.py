"""
Selects best model based on CV results and retrains it on full dataset.
Saves final_model.pkl inside models/.
Usage:
    python3 -m src.train_final_models
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet

from src.feature_engineering import remove_outliers, advanced_feature_engineering
from src.utils import advanced_preprocess_data


def load_best_model_name():
    with open("results/cv_results.json", "r") as f:
        cv = json.load(f)

    # cv = {"Ridge": [mean, std], "ElasticNet": [...], "XGBoost_optuna": [...]}

    # Pretvaramo u dict: naziv -> mean RMSE
    means = {model: score[0] for model, score in cv.items()}

    # Najbolji model je onaj sa NAJMANJIM RMSE
    best_model = min(means, key=means.get)

    print(f"🏆 Najbolji model prema CV je: {best_model}, RMSE={means[best_model]:.5f}")
    return best_model


def load_model_file(best_model_name):
    if best_model_name == "ElasticNet":
        path = "models/elasticnet_model.pkl"
        model = joblib.load(path)
        return model

    if best_model_name == "XGBoost_optuna":
        path = "models/xgboost_optuna.pkl"
        model = joblib.load(path)
        return model

    if best_model_name == "Ridge":
        raise Exception("Nema Ridge model fajla — nije treniran.")

    raise Exception(f"Nepoznat model: {best_model_name}")


def main():

    print("📌 Učitavanje najboljeg modela...")
    best_model_name = load_best_model_name()

    print("📌 Učitavanje model fajla...")
    model = load_model_file(best_model_name)

    print("📌 Učitavanje trening skupa i feature engineering...")
    df = pd.read_csv("data/train.csv")
    df = remove_outliers(df)
    df = advanced_feature_engineering(df)

    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    X = advanced_preprocess_data(X)
    X = X.fillna(0)

    print("📌 Ponovno treniranje najboljeg modela na celom datasetu...")
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/final_model.pkl")

    print("\n🎯 FINALNI MODEL SAČUVAN: models/final_model.pkl")
    print(f"Koristi se model: {best_model_name}")


if __name__ == "__main__":
    main()
