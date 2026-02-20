# House Prices – Advanced Regression Techniques

Projektni zadatak iz predmeta  
**Metode istraživanja i eksploatacije podataka**  
Fakultet tehničkih nauka

## 📌 Opis projekta

Ovaj projekat implementira kompletan machine learning pipeline za regresioni zadatak predviđanja tržišne cene kuća (SalePrice) na osnovu numeričkih i kategorijskih karakteristika objekta.

Projekat je razvijen u okviru Kaggle takmičenja:
House Prices – Advanced Regression Techniques

## ⚙️ Tehnologije

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- Optuna
- matplotlib / seaborn
- ReportLab

## 📊 Modeli

- Ridge Regression
- ElasticNet
- XGBoost
- Optimizacija hiperparametara (Optuna)

Evaluacija se vrši pomoću K-Fold cross-validation i RMSE metrike nad log-transformisanom ciljnom promenljivom.

## 📁 Struktura projekta
data/ # Ulazni skupovi podataka
src/ # Izvorni kod (pipeline)
models/ # Sačuvani modeli
results/ # Grafici, CV rezultati, PDF izveštaj


## 🚀 Pokretanje projekta
pip install -r requirements.txt
python3 -m src.train_advanced
python3 -m src.train_search --n_trials 30
python3 -m src.train_final_models
python3 -m src.predict_final
python3 -m src.visualize
python3 -m src.generate_report


## 📈 Kaggle rezultat

Screenshot rezultata nalazi se u PDF dokumentaciji projekta.

---

Autor: Dunja Cvjetinović (IT2/2020)  
Godina: 2026
