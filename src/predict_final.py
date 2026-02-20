"""
Generates final Kaggle submission using models/final_model.pkl.
Usage:
    python3 -m src.predict_final
"""

import pandas as pd
import numpy as np
import joblib
import os

from src.feature_engineering import advanced_feature_engineering
from src.utils import advanced_preprocess_data


def main():

    print("📌 Učitavanje test skupa...")
    test = pd.read_csv("data/test.csv")
    test_fe = advanced_feature_engineering(test)

    print("📌 Učitavanje train skupa (radi poravnanja kolona)...")
    train = pd.read_csv("data/train.csv")
    train_fe = advanced_feature_engineering(train)

    X_train = train_fe.drop(columns=["SalePrice"])
    X_train = advanced_preprocess_data(X_train)
    X_train = X_train.fillna(0)

    X_test = advanced_preprocess_data(test_fe)

    # --- Poravnanje kolona ---
    missing_cols = set(X_train.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_train.columns)

    # Force-remove Ames dataset ghost column
    if "RoofMatl_CompShg" in X_test.columns:
        print("⚠️ Uklanjam problematičnu kolonu RoofMatl_CompShg...")
        X_test = X_test.drop(columns=["RoofMatl_CompShg"])


    for col in missing_cols:
        X_test[col] = 0

    if extra_cols:
        print(f"⚠️ Uklanjam {len(extra_cols)} kolona koje nisu bile u trainu.")
        X_test = X_test.drop(columns=list(extra_cols))

    X_test = X_test[X_train.columns]
    X_test = X_test.fillna(0)

    # FINAL SAFETY CHECK — remove ghost column if it exists
    if "RoofMatl_CompShg" in X_test.columns:
        print("⚠️ FINAL REMOVE: RoofMatl_CompShg")
        X_test = X_test.drop(columns=["RoofMatl_CompShg"])


    print(f"📌 Kolone poravnate. Test shape = {X_test.shape}")

    # --- Učitavanje finalnog modela ---
    print("📌 Učitavanje final_model.pkl ...")
    model = joblib.load("models/final_model.pkl")

    # --- Predikcije ---
    print("📌 Generisanje predikcija...")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    # --- Snimanje submission fajla ---
    os.makedirs("results", exist_ok=True)
    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": preds
    })
    submission.to_csv("results/submission_final.csv", index=False)

    print("\n🎯 FINALNI SUBMISSION KREIRAN:")
    print("   ➝ results/submission_final.csv")
    print("📤 Uploaduj ga na Kaggle leaderboard!")
    

if __name__ == "__main__":
    main()
