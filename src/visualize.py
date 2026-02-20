import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor
from src.feature_engineering import advanced_feature_engineering
from src.utils import advanced_preprocess_data


train = pd.read_csv("data/train.csv")
train_fe = advanced_feature_engineering(train)


plt.figure(figsize=(8, 5))
sns.histplot(train_fe["SalePrice"], kde=True, color="skyblue", bins=40)
plt.title("Distribucija cena kuća", fontsize=14)
plt.xlabel("Cena kuće")
plt.ylabel("Frekvencija")
plt.tight_layout()
plt.savefig("results/plot_saleprice_distribution.png")
plt.close()


numeric_df = train_fe.select_dtypes(include=[np.number])

corr = numeric_df.corr()["SalePrice"].sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 6))
sns.barplot(x=corr.values, y=corr.index, palette="viridis")
plt.title("Top 20 najviše koreliranih feature-a sa cenom", fontsize=14)
plt.xlabel("Korelacija")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("results/plot_top_correlations.png")
plt.close()


plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr().iloc[:20, :20], cmap="coolwarm", annot=False)
plt.title("Heatmap korelacija (prvih 20 feature-a)", fontsize=14)
plt.tight_layout()
plt.savefig("results/plot_heatmap.png")
plt.close()


try:
    model = XGBRegressor()
    model.load_model("models/xgboost_final.json")
    importance = model.get_booster().get_score(importance_type="weight")
    importance_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Importance": list(importance.values())
    }).sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="mako")
    plt.title("Top 15 najvažnijih feature-a (XGBoost)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/plot_feature_importance.png")
    plt.close()
except Exception:
    print("⚠️ XGBoost model nije pronađen, preskačem feature importance graf.")

print("🎨 Svi grafovi su uspešno kreirani u folderu 'results/' ✅")
