# House Prices – Advanced Regression Techniques (Kaggle)

Ovaj projekat rešava Kaggle zadatak **House Prices: Advanced Regression Techniques** (predikcija `SalePrice` na osnovu karakteristika kuće).

## Struktura projekta

- `data/` – ulazni podaci (`train.csv`, `test.csv`, `sample_submission.csv`)
- `src/` – kod (preprocesiranje, feature engineering, treniranje, predikcija, vizualizacije, generisanje PDF dokumentacije)
- `models/` – sačuvani modeli (`.pkl`, `.json`)
- `results/` – izlazni fajlovi (submission `.csv`, grafici, `cv_results.json`, PDF izveštaj)

## Instalacija

Preporuka: virtuelno okruženje (venv/conda), pa:

```bash
pip install -r requirements.txt
```

## Pokretanje (primer)

Iz root foldera projekta:

```bash
# 1) Treniranje modela + cross-validation metrike
python3 -m src.train_advanced

# 2) (Opcionalno) Hyperparameter search (Optuna) i upis cv rezultata
python3 -m src.train_search --n_trials 30

# 3) Treniraj najbolji finalni model na celom train skupu (na osnovu cv_results.json)
python3 -m src.train_final_models

# 4) Generiši finalni submission fajl
python3 -m src.predict_final

# 5) Generiši vizualizacije
python3 -m src.visualize

# 6) Generiši PDF dokumentaciju/izveštaj
python3 -m src.generate_report
```

## Rezultati

- CV metrike se čuvaju u `results/cv_results.json`
- Submission fajlovi se čuvaju u `results/` (npr. `submission_final.csv`)
- PDF dokumentacija se generiše u `results/House_Prices_Report.pdf`

## Napomena

Ako ne koristiš LightGBM/CatBoost, možeš ih izbaciti iz `requirements.txt`.
