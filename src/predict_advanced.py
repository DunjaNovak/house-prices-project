import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, ElasticNet

from src.feature_engineering import advanced_feature_engineering
from src.utils import advanced_preprocess_data

# ====== 1. Učitaj test i train podatke ======
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

# ====== 2. Feature engineering ======
test_fe = advanced_feature_engineering(test)
train_fe = advanced_feature_engineering(train)

# ====== 3. Priprema train skupa ======
X_train = train_fe.drop(columns=["SalePrice"])
X_train = advanced_preprocess_data(X_train)
X_train = X_train.fillna(0)

# ====== 4. Priprema test skupa ======
X_test = advanced_preprocess_data(test_fe)

# ✅ Usklađivanje kolona između train i test skupa
missing_cols = set(X_train.columns) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(X_train.columns)

# Dodaj nedostajuće kolone u test i popuni ih nulama
for col in missing_cols:
    X_test[col] = 0

# Ukloni kolone koje nisu postojale u train skupu
if extra_cols:
    print(f"⚠️ Uklanjam {len(extra_cols)} kolona koje nisu bile u trainu: {list(extra_cols)[:5]}...")
    X_test = X_test.drop(columns=list(extra_cols))

# Reorganizuj kolone da idu istim redosledom
X_test = X_test[X_train.columns]
X_test = X_test.fillna(0)

print(f"✅ Train features: {len(X_train.columns)}, Test features: {X_test.shape[1]}")

# ====== 5. Učitaj najbolji model ======
try:
    model = XGBRegressor()
    model.load_model("models/xgboost_final.json")
    model_type = "xgboost"
except Exception:
    try:
        model = joblib.load("models/elasticnet_model.pkl")
        model_type = "elasticnet"
    except Exception:
        model = joblib.load("models/ridge_model.pkl")
        model_type = "ridge"

print(f"✅ Učitan model tipa: {model_type}")

# 🔧 Ukloni problematičnu kolonu ako slučajno još postoji
if "RoofMatl_CompShg" in X_test.columns:
    print("⚠️ Uklanjam kolonu RoofMatl_CompShg koja nije postojala tokom treniranja.")
    X_test = X_test.drop(columns=["RoofMatl_CompShg"])

# ====== 6. Predikcija ======
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)  # vraća iz log skale u normalne cene

# ====== 7. Kreiranje submission fajla ======
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds
})
submission.to_csv("results/submission_advanced.csv", index=False)
print("🎯 Submission fajl kreiran: results/submission_advanced.csv")
