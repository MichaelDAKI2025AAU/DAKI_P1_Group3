import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =========================================================
# Load data
# =========================================================
df = pd.read_csv("C:\\Users\\thoma\\OneDrive\\Dokumenter\\DAKI\\P1\\Covid Data.csv")

# =========================================================
# Define death variable DIED
# =========================================================
df["DATE_DIED"] = df["DATE_DIED"].astype(str)
df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

# =========================================================
# Features / kolonner vi bruker
# =========================================================
feature_cols = [
    "AGE",
    "SEX",
    "DIABETES",
    "HIPERTENSION",
    "OBESITY",
    "COPD",
    "ASTHMA",
    "CARDIOVASCULAR",
    "RENAL_CHRONIC",
    "INMSUPR",
    "TOBACCO",
    "OTHER_DISEASE"
]

binary_columns = [c for c in feature_cols if c != "AGE"]

# =========================================================
# Riktig håndtering av 97/98/99 + alder
# =========================================================

# 1) Fjern 97/98/99 KUN i binære variabler (+ evt. CLASIFFICATION_FINAL)
cols_with_missing_codes = binary_columns + ["PREGNANT", "CLASIFFICATION_FINAL"]
for col in cols_with_missing_codes:
    if col in df.columns:
        df[col] = df[col].replace({97: np.nan, 98: np.nan, 99: np.nan})

# 2) Binære kolonner: 1/2 -> 1/0
for col in binary_columns:
    df[col] = df[col].replace({1: 1, 2: 0})

# PREGNANT som ekstra binær, hvis den brukes senere
if "PREGNANT" in df.columns:
    df["PREGNANT"] = df["PREGNANT"].replace({1: 1, 2: 0})
    df["PREGNANT"] = df["PREGNANT"].fillna(0)

# 3) Alder som numerisk + trunkering ved 110 år
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df.loc[df["AGE"] > 110, "AGE"] = np.nan

# ---------------------------------------------------------
# FUNCTION 1: Fit LR model and return results
# ---------------------------------------------------------

def train_logistic_regression(name, data):
    print(f"\n\n============================")
    print(f"     {name.upper()} - LOGISTISK REGRESJON")
    print(f"============================\n")

    # Drop NA
    mask = data[feature_cols + ["DIED"]].notna().all(axis=1)
    clean = data[mask]

    X = clean[feature_cols]
    y = clean["DIED"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline (skalering av AGE + LR)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)

    # Predictions
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Evaluation
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Statsmodels OR (uten skalering, på hele clean datasett)
    print("\n--- Odds Ratio (statsmodels) ---")
    X_sm = sm.add_constant(X)
    logit = sm.Logit(y, X_sm)
    result = logit.fit()
    print(result.summary())

    params = result.params
    conf = result.conf_int()
    conf["OR"] = params
    conf.columns = ["2.5%", "97.5%", "OR"]
    odds_ratios = np.exp(conf)
    print(odds_ratios.sort_values(by="OR", ascending=False))

    # Forest plot
    or_df = odds_ratios.drop("const").sort_values(by="OR", ascending=False)

    variables = or_df.index
    or_values = or_df["OR"]
    lower = or_df["2.5%"]
    upper = or_df["97.5%"]

    plt.figure(figsize=(8, 6))
    plt.scatter(or_values, variables, color="black")
    plt.hlines(variables, lower, upper, color="black")
    plt.axvline(1, color="red", linestyle="--")
    plt.xscale("log")
    plt.title(f"Forest plot: {name} – Odds Ratio (95% CI)")
    plt.xlabel("Odds Ratio (log-skala)")
    plt.tight_layout()
    plt.show()

    return odds_ratios, result


# ---------------------------------------------------------
# MODEL A: Hele datasettet (alle rader, alle CLASIFFICATION_FINAL)
# ---------------------------------------------------------
odds_all, model_all = train_logistic_regression(
    "Modell A – Hele datasettet",
    df
)

# ---------------------------------------------------------
# MODEL B: Kun COVID-positive (CLASIFFICATION_FINAL = 1, 2, 3)
# ---------------------------------------------------------
if "CLASIFFICATION_FINAL" in df.columns:
    df_pos = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].copy()

    odds_pos, model_pos = train_logistic_regression(
        "Modell B – COVID-positive",
        df_pos
    )

#test