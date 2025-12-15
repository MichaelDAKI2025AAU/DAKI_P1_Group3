import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve)

# Dataforberedelse
def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # 1) BINÆRE VARIABLER – fjern kun 97/98/99 HER (ikke i AGE)
    binary_cols = ["SEX", "DIABETES", "HIPERTENSION", "OBESITY", 
                   "COPD", "ASTHMA","CARDIOVASCULAR", "RENAL_CHRONIC", 
                   "INMSUPR", "TOBACCO", "OTHER_DISEASE", "PREGNANT",]
    df[binary_cols] = df[binary_cols].replace({97: np.nan, 98: np.nan, 99: np.nan})

    # 2) FILTRER KUN COVID-SMITTEDE (1,2,3)
    df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

    # 3) DØDSVARIABEL (DIED)
    df["DATE_DIED"] = df["DATE_DIED"].astype(str)
    df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

    # 4) BINARISER 1/2 → 1/0
    for col in binary_cols:
        df[col] = df[col].replace({1: 1, 2: 0})
    df["PREGNANT"] = df["PREGNANT"].fillna(0)

    # 5) HÅNDTER ALDER (INGEN fjernelse af 97/98/99)
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

    # 8) ENDLIGE FEATURE-LISTER
    categorical = binary_cols + age_dummies + comorb_dummies
    numeric = ["AGE"]
    feature_names = numeric + categorical

    # 9) FJERN RÆKKER MED MANGLENDE FEATURES
    mask = df[categorical + numeric + ["DIED"]].notna().all(axis=1)
    clean = df[mask]

    # 10) SPLIT PÅ RÅ DATA
    X = clean[categorical + numeric]
    y = clean["DIED"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # 11) UNDERSAMPLING KUN PÅ TRÆNINGSDATA
    train_df = pd.concat([X_train, y_train], axis=1)

    df_male_dead     = train_df[(train_df.SEX==1) & (train_df.DIED==1)]
    df_male_alive    = train_df[(train_df.SEX==1) & (train_df.DIED==0)]
    df_female_dead   = train_df[(train_df.SEX==0) & (train_df.DIED==1)]
    df_female_alive  = train_df[(train_df.SEX==0) & (train_df.DIED==0)]

    n = min(len(df_male_dead), len(df_male_alive),
            len(df_female_dead), len(df_female_alive))

    train_balanced = pd.concat([
        df_male_dead.sample(n, random_state=42),
        df_male_alive.sample(n, random_state=42),
        df_female_dead.sample(n, random_state=42),
        df_female_alive.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42)

    # Opdater træningssæt med undersamplede data
    X_train = train_balanced[categorical + numeric]
    y_train = train_balanced["DIED"]

    # RETURN – det endelige output
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, numeric

def build_preprocessor(numeric_cols):
    #Bygger én samlet ColumnTransformer, som alle modeller bruger.
    #Numeriske kolonner skaleres, resten passeres igennem.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough")
    return preprocessor

# Base model-klasse
class BaseModel:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_threshold = 0.5
        self.metrics_ = None
        self.y_val_prob_ = None
        self.y_test_prob_ = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def find_optimal_threshold(self, y_true, y_prob):
        #Finder optimal threshold for modellen vha. Youden-indekset:
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t = 0.5
        best_youden = -1.0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()
            sensitivity = recall_score(y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            youden = sensitivity + specificity - 1

            if youden > best_youden:
                best_youden = youden
                best_t = t
        self.best_threshold = best_t
        return best_t, best_youden

    def evaluate_on_test(self, y_true, y_prob):
        #Evaluerer modellen på test-sættet med den allerede fundne best_threshold.
        #Returnerer et metrics-dict og konfusionsmatrix.
        t = self.best_threshold
        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = 2 * prec * sensitivity / (prec + sensitivity + 1e-9)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        metrics = {
            "Model": self.name,
            "AP": ap,
            "AUC": auc,
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": sensitivity,
            "Specificity": specificity,
            "F1": f1,
            "Youden": sensitivity + specificity - 1,
            "ConfusionMatrix": np.array([[tn, fp], [fn, tp]]),
            "ClassificationReport": classification_report(
                y_true, y_pred, digits=3, zero_division=0),}
        self.metrics_ = metrics
        return metrics

    def get_feature_importances(self, feature_names):
        #Overloades i de modeller, der har feature importance.
        return None

# Model-specifikke klasser
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(name="Logistic Regression")
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs")

    def get_feature_importances(self, feature_names):
        # Koef for binær klassifikation er shape (1, n_features)
        coefs = self.model.coef_[0]

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefs),   # typisk importance
            "Coefficient": coefs           # original signed weight
        }).sort_values("Importance", ascending=False)

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(name="Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1)

    def get_feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

class CatBoostModel(BaseModel):
    def __init__(self, scale_pos_weight):
        super().__init__(name="CatBoost")
        self.model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=False)

    def get_feature_importances(self, feature_names):
        importances = self.model.get_feature_importance()
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

# ModelManager – kører alle modeller og laver plots/tabeller
class ModelManager:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 feature_names, scale_pos_weight_cat):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_names

        # Rækkefølge: LR → RF → CatBoost
        self.models = [
            LogisticRegressionModel(),
            RandomForestModel(),
            CatBoostModel(scale_pos_weight=scale_pos_weight_cat),]

        self.results = []          # liste af metrics-dicts
        self.y_test_probas = {}    # til ROC-plot
        self.conf_matrices = {}    # til confusion-plots
        self.feature_importances = {}  # RF + CatBoost

    def run_all(self):
        for model in self.models:
            print("\n" + "=" * 60)
            print(f"{model.name.upper()}")
            print("=" * 60)

            # 1) Træn på train
            model.fit(self.X_train, self.y_train)

            # 2) Threshold-optimering på val via Youden
            y_val_prob = model.predict_proba(self.X_val)
            model.y_val_prob_ = y_val_prob
            best_t, best_youden = model.find_optimal_threshold(self.y_val, y_val_prob)
            print(f"Optimal threshold (Youden) på val-sæt for {model.name}: "
                  f"{best_t:.3f} (Youden = {best_youden:.3f})")

            # 3) Evaluer på test-sættet med denne threshold
            y_test_prob = model.predict_proba(self.X_test)
            model.y_test_prob_ = y_test_prob
            metrics = model.evaluate_on_test(self.y_test, y_test_prob)

            # Print nøglemetrics
            print("\n--- Nøglemetrics (test-sæt) ---")
            for k in ["AP", "AUC", "Threshold", "Accuracy", "Precision",
                      "Recall", "Specificity", "F1", "Youden"]:
                print(f"{k:12s}: {metrics[k]:.3f}")

            print("\nKonfusionsmatrix (test-sæt):")
            print(metrics["ConfusionMatrix"])

            print("\nClassification report (test-sæt):")
            print(metrics["ClassificationReport"])

            # Gem til senere opsamling
            self.results.append({k: metrics[k] for k in metrics if k != "ClassificationReport"})
            self.y_test_probas[model.name] = y_test_prob
            self.conf_matrices[model.name] = metrics["ConfusionMatrix"]

            # Feature importance hvor muligt
            fi = model.get_feature_importances(self.feature_names)
            if fi is not None:
                self.feature_importances[model.name] = fi

    def create_summary_table(self):
        #Samlet tabel med nøgletal for alle modeller (test-sæt).
        df_res = pd.DataFrame(self.results)
        # Sortér evt. efter AUC
        df_res = df_res[[
            "Model", "AP", "AUC", "Threshold", "Accuracy",
            "Precision", "Recall", "Specificity", "F1", "Youden"]]
        return df_res

    def plot_roc_curves(self):
        #ROC-kurver for alle modeller i én figur (test-sæt).
        plt.figure(figsize=(8, 6))

        for model in self.models:
            name = model.name
            y_prob = self.y_test_probas[name]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = roc_auc_score(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-kurver – test-sæt")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self):
        #Confusion matrices for alle tre modeller (test-sæt) med Matplotlib.
        n_models = len(self.models)
        plt.figure(figsize=(5 * n_models, 4))

        for i, model in enumerate(self.models, start=1):
            name = model.name
            cm = self.conf_matrices[name]

            plt.subplot(1, n_models, i)
            plt.imshow(cm, interpolation="nearest", cmap="coolwarm")
            plt.title(name)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
            plt.yticks(tick_marks, ["True 0", "True 1"])

            # Tal inde i felterne
            thresh = cm.max() / 2.0
            for r in range(2):
                for c in range(2):
                    plt.text(
                        c, r, format(cm[r, c], "d"),
                        ha="center", va="center",
                        color="white" if cm[r, c] > thresh else "black",
                        fontsize = 16)

            plt.ylabel("Sand klasse")
            plt.xlabel("Forudsagt klasse")

        plt.suptitle("Konfusionsmatricer – test-sæt")
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curves(self):
        #Precision-Recall curves for all models (test-set).
        plt.figure(figsize=(8, 6))
        for model in self.models:
            name = model.name
            y_prob = self.y_test_probas[name]
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            ap = average_precision_score(self.y_test, y_prob)

            plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

        plt.xlabel("Recall (TPR)")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curves – test-sæt")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, top_n=20):
        #Feature importance plots for Random Forest og CatBoost.
        for model_name, fi_df in self.feature_importances.items():
            top = fi_df.head(top_n)
            plt.figure(figsize=(8, 6))
            plt.barh(top["Feature"], top["Importance"])
            plt.gca().invert_yaxis()
            plt.title(f"{model_name} – Top {top_n} vigtigste features")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

# Hovedprogram – alt køres herfra
if __name__ == "__main__":
    # 1) Forbered data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, numeric = prepare_data(
        "CovidData.csv")

    # Fælles preprocessor til alle modeller
    preprocessor = build_preprocessor(numeric_cols=numeric)

    # Fit på train, transformér alle sæt
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # scale_pos_weight til CatBoost
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight_cat = n_neg / n_pos

    # 2) Opret ModelManager og kør modeller i rækkefølge:
    #    LR → RF → CatBoost
    manager = ModelManager(
        X_train=X_train_proc,
        y_train=y_train.values,
        X_val=X_val_proc,
        y_val=y_val.values,
        X_test=X_test_proc,
        y_test=y_test.values,
        feature_names=feature_names,
        scale_pos_weight_cat=scale_pos_weight_cat)
    manager.run_all()

    # 3) Samlet tabel + grafer (ALT DETTE ER TIL SIDST)
    summary_df = manager.create_summary_table()
    print("\n" + "=" * 60)
    print("SAMLET RESULTATTABEL (test-sæt)")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ROC-kurver
    manager.plot_roc_curves()
    # Confusion matrices
    manager.plot_confusion_matrices()
    # Precision–Recall curves
    manager.plot_precision_recall_curves()
    # Feature importance (RF + CatBoost)
    manager.plot_feature_importances(top_n=20)