import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,)
from sklearn.isotonic import IsotonicRegression
import joblib
from catboost import CatBoostClassifier

# 1. Dataforberedelse

def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # 1) BINÆRE VARIABLER – kun fjern 97/98/99 HER (ikke AGE)
    binary_cols = [
        "SEX", "DIABETES", "HIPERTENSION", "OBESITY", "COPD", "ASTHMA",
        "CARDIOVASCULAR", "RENAL_CHRONIC", "INMSUPR", "TOBACCO",
        "OTHER_DISEASE", "PREGNANT"]
    df[binary_cols] = df[binary_cols].replace({97: np.nan, 98: np.nan, 99: np.nan})

    # 2) FILTRER KUN COVID-SMITTEDE (1,2,3)
    df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

    # 3) DØDSVARIABEL
    df["DATE_DIED"] = df["DATE_DIED"].astype(str)
    df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

    # 4) BINARISERING (1/2 → 1/0)
    df["SEX"] = df["SEX"].replace({1: 0, 2: 1})  # 0=kvinde, 1=mand

    for col in binary_cols:
        df[col] = df[col].replace({1: 1, 2: 0})
    df["PREGNANT"] = df["PREGNANT"].fillna(0)

    # 5) AGE – behold 97/98/99, kun clip
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["AGE"] = df["AGE"].clip(upper=110)

    # 6) ALDERSKATEGORIER + DUMMIES
    df["AGE_CAT"] = pd.cut(
        df["AGE"],
        bins=[0, 50, 60, 70, 80, 120],
        labels=["<50", "50-59", "60-69", "70-79", "80+"],
        right=False)
    df = pd.get_dummies(df, columns=["AGE_CAT"], drop_first=True)
    age_dummies = [c for c in df.columns if c.startswith("AGE_CAT_")]

    # 7) MULTIMORBIDITET
    df["COMORB_COUNT"] = df[binary_cols].sum(axis=1)
    df["COMORB_CAT"] = df["COMORB_COUNT"].clip(upper=6).astype("category")
    df = pd.get_dummies(df, columns=["COMORB_CAT"], drop_first=True)
    comorb_dummies = [c for c in df.columns if c.startswith("COMORB_CAT_")]

    # 8) FØR UNDERSAMPLING: DROP NA
    categorical = binary_cols + age_dummies + comorb_dummies
    numeric = ["AGE"]
    base_features = numeric + categorical

    mask = df[base_features + ["DIED"]].notna().all(axis=1)
    clean = df[mask].copy()

    # 9) SPLIT (RAW DATA)
    X = clean[base_features]
    y = clean["DIED"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # 10) UNDERSAMPLING KUN PÅ TRAIN
    train_df = pd.concat([X_train, y_train], axis=1)

    df_male_dead = train_df[(train_df.SEX == 1) & (train_df.DIED == 1)]
    df_male_alive = train_df[(train_df.SEX == 1) & (train_df.DIED == 0)]
    df_female_dead = train_df[(train_df.SEX == 0) & (train_df.DIED == 1)]
    df_female_alive = train_df[(train_df.SEX == 0) & (train_df.DIED == 0)]

    n = min(len(df_male_dead), len(df_male_alive),
            len(df_female_dead), len(df_female_alive))

    train_balanced = pd.concat([
        df_male_dead.sample(n, random_state=42),
        df_male_alive.sample(n, random_state=42),
        df_female_dead.sample(n, random_state=42),
        df_female_alive.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42)

    # Opdater X_train, y_train
    X_train = train_balanced[base_features]
    y_train = train_balanced["DIED"]

    # 11) INTERAKTIONER (V4-FEATURES)
    interaction_features = []

    X_train["AGE_X_SEX"] = X_train["AGE"] * X_train["SEX"]
    X_val["AGE_X_SEX"] = X_val["AGE"] * X_val["SEX"]
    X_test["AGE_X_SEX"] = X_test["AGE"] * X_test["SEX"]
    interaction_features.append("AGE_X_SEX")

    for col in binary_cols:
        feat = f"AGE_X_{col}"
        X_train[feat] = X_train["AGE"] * X_train[col]
        X_val[feat] = X_val["AGE"] * X_val[col]
        X_test[feat] = X_test["AGE"] * X_test[col]
        interaction_features.append(feat)

    # 12) LAV DEN ENDELIGE FEATURELISTE TIL MODELLEN
    feature_cols = base_features + interaction_features

    # RETURN
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

# 2. Hjælpefunktioner: threshold & evaluering

def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_youden = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1

        if youden > best_youden:
            best_youden = youden
            best_t = t
    return best_t, best_youden


def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = 2 * prec * sens / (prec + sens + 1e-9)
    auc = roc_auc_score(y_true, y_prob)
    youden = sens + spec - 1

    print("\n===== Test-metrics for GUI v4 (kalibreret) =====")
    print(f"AUC        : {auc:.3f}")
    print(f"Threshold  : {threshold:.3f}")
    print(f"Accuracy   : {acc:.3f}")
    print(f"Precision  : {prec:.3f}")
    print(f"Recall     : {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    print(f"F1-score   : {f1:.3f}")
    print(f"Youden     : {youden:.3f}")

    print("\nKonfusionsmatrix (test):")
    print(np.array([[tn, fp], [fn, tp]]))

    print("\nClassification report (test):")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    metrics = {
        "AUC": auc,
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": sens,
        "Specificity": spec,
        "F1": f1,
        "Youden": youden,
        "ConfusionMatrix": np.array([[tn, fp], [fn, tp]]),}
    return metrics

# 3. Hovedprogram – træning + kalibrering

def main():
    csv_path = "CovidData.csv"
    (   X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_cols,
    ) = prepare_data(csv_path)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # CatBoost-model (moderat regulariseret)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False,)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,)

    # AUC på val (ukalibreret)
    y_val_raw = model.predict_proba(X_val)[:, 1]
    val_auc_raw = roc_auc_score(y_val, y_val_raw)
    print(f"\nAUC (validation, rå model): {val_auc_raw:.3f}")

    # Isotonic calibration på VALIDATION-sættet
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_val_raw, y_val)

    y_val_cal = iso.predict(y_val_raw)
    val_auc_cal = roc_auc_score(y_val, y_val_cal)
    print(f"AUC (validation, kalibreret): {val_auc_cal:.3f}")

    # Optimal threshold på kalibreret val-prob
    best_t, best_youden = find_optimal_threshold(y_val, y_val_cal)
    print(
        f"Optimal threshold (Youden, val – kalibreret): "
        f"{best_t:.3f} (Youden = {best_youden:.3f})")

    # Test-evaluering med kalibreret risiko
    y_test_raw = model.predict_proba(X_test)[:, 1]
    y_test_cal = iso.predict(y_test_raw)

    _ = evaluate_model(y_test, y_test_cal, threshold=best_t)

    # Gem model + kalibrator til GUI'en
    model_path = "catboost_covid_gui_v4.cbm"
    calib_path = "calibrator_gui_v4.pkl"

    model.save_model(model_path)
    joblib.dump(iso, calib_path)

    print(f"\n[INFO] GUI-model gemt som '{model_path}'")
    print(f"[INFO] Kalibrator gemt som '{calib_path}'")

    # Print feature-ordenen til GUI
    print("\n===== FEATURE-ORDEN TIL GUI v4 =====")
    for i, col in enumerate(feature_cols):
        print(f"{i:2d}: {col}")

    print(
        "\nNB: GUI'en skal konstruere et feature-vector i PRÆCIS denne rækkefølge.\n"
        "Basale inputfelter (AGE, SEX + sygdomme) omsættes internt til:\n"
        "- AGE\n"
        "- SEX (1=mand, 0=kvinde)\n"
        "- 11 komorbiditeter\n"
        "- AGE_CAT-dummies (ud fra AGE)\n"
        "- COMORB_CAT-dummies (ud fra antal komorbiditeter)\n"
        "- Interaktions-features (AGE_X_...)\n")

if __name__ == "__main__":
    main()
