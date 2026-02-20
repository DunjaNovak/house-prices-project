import pandas as pd
import numpy as np
from xgboost import XGBRegressor

test = pd.read_csv("data/test.csv")

def basic_feature_engineering(df):
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    return df

test = basic_feature_engineering(test)

train = pd.read_csv("data/train.csv")
train = basic_feature_engineering(train)
X_train = train.select_dtypes(include=["int64", "float64"]).drop("SalePrice", axis=1)
cols = X_train.columns

for col in cols:
    if col not in test.columns:
        test[col] = 0
test = test[cols]
test = test.fillna(test.median())

model = XGBRegressor()
model.load_model("models/xgboost_model.json")

preds_log = model.predict(test)
preds = np.expm1(preds_log)  

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds
})

submission.to_csv("results/submission.csv", index=False)
print("✅ Submission fajl kreiran: results/submission.csv")
