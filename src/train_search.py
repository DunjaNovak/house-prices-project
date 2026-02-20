"""
Trains multiple models with hyperparameter search (Optuna for GBMs, Randomized for linear).
Saves best models and CV metrics into results/.
Usage:
    python3 -m src.train_search --n_trials 30
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
import joblib

# GBMs
import optuna
from xgboost import XGBRegressor
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None

from src.feature_engineering import remove_outliers, advanced_feature_engineering
from src.utils import advanced_preprocess_data

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

def cv_score(model, X, y, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = -1 * cross_val_score(model, X, y, scoring=rmse_scorer, cv=cv, n_jobs=-1)
    return scores.mean(), scores.std()

def optuna_xgboost(X, y, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        score, _ = cv_score(model, X, y, n_splits=5)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optuna_lgb(X, y, n_trials=30):
    if lgb is None:
        return None
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
        score, _ = cv_score(model, X, y, n_splits=5)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optuna_catboost(X, y, n_trials=30):
    if CatBoostRegressor is None:
        return None
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
        model = CatBoostRegressor(**params, random_state=42, verbose=False)
        score, _ = cv_score(model, X, y, n_splits=5)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("data/train.csv")
    df = remove_outliers(df)
    df = advanced_feature_engineering(df)
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])
    X = advanced_preprocess_data(X)
    X = X.fillna(0)

    results = {}

    # Linear models quick CV
    print("-> Evaluating Ridge and ElasticNet with default params")
    ridge = Ridge(alpha=10.0, random_state=42)
    enr = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42, max_iter=5000)
    results["Ridge"] = cv_score(ridge, X, y, n_splits=5)
    results["ElasticNet"] = cv_score(enr, X, y, n_splits=5)

    # XGBoost Optuna
    print("-> Tuning XGBoost with Optuna")
    xgb_best = optuna_xgboost(X, y, n_trials=args.n_trials)
    print("XGBoost best params:", xgb_best)
    xgb_model = XGBRegressor(**xgb_best, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, "models/xgboost_optuna.pkl")
    results["XGBoost_optuna"] = cv_score(xgb_model, X, y, n_splits=5)

    # LightGBM 
    if lgb is not None:
        print("-> Tuning LightGBM with Optuna")
        lgb_best = optuna_lgb(X, y, n_trials=args.n_trials)
        print("LightGBM best params:", lgb_best)
        lgb_model = lgb.LGBMRegressor(**lgb_best, random_state=42, n_jobs=-1)
        lgb_model.fit(X, y)
        joblib.dump(lgb_model, "models/lgbm_optuna.pkl")
        results["LGBM_optuna"] = cv_score(lgb_model, X, y, n_splits=5)

    # CatBoost 
    if CatBoostRegressor is not None:
        print("-> Tuning CatBoost with Optuna")
        cat_best = optuna_catboost(X, y, n_trials=args.n_trials)
        print("CatBoost best params:", cat_best)
        cat_model = CatBoostRegressor(**cat_best, random_state=42, verbose=False)
        cat_model.fit(X, y)
        joblib.dump(cat_model, "models/catboost_optuna.pkl")
        results["CatBoost_optuna"] = cv_score(cat_model, X, y, n_splits=5)

  
    with open("results/cv_results.json", "w") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in results.items()}, f, indent=2)

   
    print("CV results:", results)
    print("Saved models to models/ and CV summary to results/cv_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trials")
    args = parser.parse_args()
    main(args)
