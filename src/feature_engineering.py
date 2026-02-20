import pandas as pd
import numpy as np

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    return df


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_feature_engineering(df)

    # Interakcioni feature-i
    df["QualityArea"] = df["OverallQual"] * df["TotalSF"]
    df["QualityCond"] = df["OverallQual"] * df["OverallCond"]
    df["QualityBath"] = df["OverallQual"] * df["TotalBath"]

    # Kompozitni feature-i
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["TotalRooms"] = df["TotRmsAbvGrd"] + df["FullBath"] + df["HalfBath"]
    df["LivingAreaPerRoom"] = df["GrLivArea"] / df["TotalRooms"].replace(0, np.nan)

    # Izvedeni feature-i
    df["BasementFinishRatio"] = df["BsmtFinSF1"] / df["TotalBsmtSF"].replace(0, np.nan)
    df["WasRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["YearsRemodeled"] = df["YrSold"] - df["YearRemodAdd"]
    df["RecentRemodel"] = (df["YearsRemodeled"] < 5).astype(int)

    # Skorovi (1–5)
    df["GarageScore"] = df["GarageArea"] * df["GarageQual"].map({
        'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1
    }).fillna(0)

    df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"].map({
        'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1
    }).fillna(0)

    # Polinomijalni feature-i
    df["TotalSF2"] = df["TotalSF"] ** 2
    df["OverallQual2"] = df["OverallQual"] ** 2
    df["Age2"] = df["Age"] ** 2

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["GrLivArea"] < 4000]
    df = df[df["SalePrice"] > 30000]
    return df
