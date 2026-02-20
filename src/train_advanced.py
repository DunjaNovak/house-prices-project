import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error

from src.feature_engineering import remove_outliers, advanced_feature_engineering
from src.utils import advanced_preprocess_data, scale_data

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

train = pd.read_csv("data/train.csv")
train = remove_outliers(train)
train = advanced_feature_engineering(train)

y = np.log1p(train["SalePrice"])  
X = train.drop(columns=["SalePrice"])

X = advanced_preprocess_data(X)
X = X.fillna(0)

models = {
    "Ridge": Ridge(alpha=10.0),
    "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    print(f"⏳ Treniranje modela: {name}")
    scores = -1 * cross_val_score(model, X, y, scoring=rmse_scorer, cv=cv)
    results[name] = (scores.mean(), scores.std())
    print(f"✅ {name}: RMSE={scores.mean():.5f} ± {scores.std():.5f}")

print("\n=== Rezime rezultata ===")
for name, (mean_rmse, std_rmse) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"{name}: RMSE={mean_rmse:.5f} ± {std_rmse:.5f}")

best_model_name = min(results, key=lambda x: results[x][0])
print(f"\n🏆 Najbolji model: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X, y)

import joblib

# Ako je model XGBoost, sačuvaj kao .json
if best_model_name == "XGBoost":
    best_model.save_model("models/xgboost_final.json")
# Ako je ElasticNet ili Ridge, sačuvaj kao .pkl
else:
    joblib.dump(best_model, f"models/{best_model_name.lower()}_model.pkl")

print(f"✅ Model {best_model_name} sačuvan u folderu models/")

best_model.save_model(f"models/{best_model_name.lower()}_final.json") if best_model_name == "XGBoost" else None

print("✅ Model sačuvan!")
