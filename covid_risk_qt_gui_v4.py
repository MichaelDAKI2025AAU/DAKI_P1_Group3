import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QGridLayout, QMessageBox, QCheckBox)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt
from catboost import CatBoostClassifier
import joblib

# Risikoniveau-farver
def risk_level(prob):
    if prob < 0.10:
        return "LAV risiko", QColor(0, 180, 0)
    elif prob < 0.40:
        return "MIDDEL risiko", QColor(230, 180, 0)
    else:
        return "HØJ risiko", QColor(220, 0, 0)

# GUI APP – v4 (med kalibreret risiko + interaktioner)
class CovidRiskApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load CatBoost-model + kalibrator
        self.model = CatBoostClassifier()
        self.model.load_model("catboost_covid_gui_v4.cbm")
        self.calibrator = joblib.load("calibrator_gui_v4.pkl")

        # Feature-rækkefølge SKAL matche training-scriptet v4
        self.feature_order = [
            # --- Basale features ---
            "AGE", "SEX", "DIABETES", "HIPERTENSION",
            "OBESITY", "COPD", "ASTHMA", "CARDIOVASCULAR", 
            "RENAL_CHRONIC", "INMSUPR", "TOBACCO", 
            "OTHER_DISEASE", "PREGNANT",

            # --- AGE_CAT dummies ---
            "AGE_CAT_50-59", "AGE_CAT_60-69",
            "AGE_CAT_70-79", "AGE_CAT_80+",

            # --- COMORB_CAT dummies ---
            "COMORB_CAT_1", "COMORB_CAT_2", "COMORB_CAT_3",
            "COMORB_CAT_4", "COMORB_CAT_5", "COMORB_CAT_6",

            # --- Interaktions-features ---
            "AGE_X_SEX", "AGE_X_DIABETES", "AGE_X_HIPERTENSION",
            "AGE_X_OBESITY", "AGE_X_COPD", "AGE_X_ASTHMA",
            "AGE_X_CARDIOVASCULAR", "AGE_X_RENAL_CHRONIC", "AGE_X_INMSUPR",
            "AGE_X_TOBACCO", "AGE_X_OTHER_DISEASE", "AGE_X_PREGNANT",]

        self.init_ui()

    # GUI-layouet
    def init_ui(self):
        self.setWindowTitle("COVID-19 Dødelighedsrisiko – v4 (kalibreret)")
        self.setGeometry(200, 200, 800, 550)

        layout = QGridLayout()

        title = QLabel("COVID-19 Dødelighedsrisiko")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 0, 1, 2)

        # Alder
        layout.addWidget(QLabel("Alder (år):"), 1, 0)
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Indtast alder i år (fx 65)")
        layout.addWidget(self.age_input, 1, 1)

        # Køn
        # VIGTIGT: 1 = mand, 0 = kvinde (matcher træningen)
        self.sex_checkbox = QCheckBox("Køn: Mand (Ja = mand, Nej = kvinde)")
        layout.addWidget(self.sex_checkbox, 2, 0, 1, 2)

        # Komorbiditeter
        layout.addWidget(QLabel("Komorbiditeter (Ja = til stede):"), 3, 0, 1, 2)

        self.disease_checkboxes = {}
        self.disease_labels = {
            "DIABETES": "Diabetes",
            "HIPERTENSION": "Hypertension",
            "OBESITY": "Overvægt",
            "COPD": "KOL",
            "ASTHMA": "Astma",
            "CARDIOVASCULAR": "Hjerte-kar sygdom",
            "RENAL_CHRONIC": "Nyresygdom",
            "INMSUPR": "Immunosupprimeret",
            "TOBACCO": "Ryger",
            "OTHER_DISEASE": "Anden sygdom",
            "PREGNANT": "Gravid"}

        row = 4
        for key, label in self.disease_labels.items():
            cb = QCheckBox(label)
            self.disease_checkboxes[key] = cb
            layout.addWidget(cb, row, 0, 1, 2)
            row += 1

        # Knap
        self.button = QPushButton("Beregn risiko")
        self.button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.button.clicked.connect(self.compute_risk)
        layout.addWidget(self.button, row, 0, 1, 2)

        # Resultat-label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label, row + 1, 0, 1, 2)

        self.setLayout(layout)

    # Hjælpefunktioner til features
    def age_cat_features(self, age):
        return {
            "AGE_CAT_50-59": 1 if 50 <= age < 60 else 0,
            "AGE_CAT_60-69": 1 if 60 <= age < 70 else 0,
            "AGE_CAT_70-79": 1 if 70 <= age < 80 else 0,
            "AGE_CAT_80+":   1 if age >= 80      else 0,}

    def comorbidity_category(self, disease_values):
        comorbs = [
            "DIABETES", "HIPERTENSION", "OBESITY", "COPD", "ASTHMA",
            "CARDIOVASCULAR", "RENAL_CHRONIC", "INMSUPR",
            "TOBACCO", "OTHER_DISEASE"]
        count = sum(disease_values[k] for k in comorbs)
        count = min(count, 6)  # clip 6+

        return {
            "COMORB_CAT_1": 1 if count == 1 else 0,
            "COMORB_CAT_2": 1 if count == 2 else 0,
            "COMORB_CAT_3": 1 if count == 3 else 0,
            "COMORB_CAT_4": 1 if count == 4 else 0,
            "COMORB_CAT_5": 1 if count == 5 else 0,
            "COMORB_CAT_6": 1 if count >= 6 else 0,}

    def interaction_features(self, age, sex, disease_values):
        feats = {
            "AGE_X_SEX": age * sex}
        for col in self.disease_labels.keys():
            feats[f"AGE_X_{col}"] = age * disease_values[col]
        return feats

    # Beregning af risiko
    def compute_risk(self):
        # Alder
        try:
            age = float(self.age_input.text())
        except ValueError:
            QMessageBox.critical(self, "Fejl", "Alder skal være et gyldigt tal.")
            return
        if not (0 <= age <= 120):
            QMessageBox.critical(self, "Fejl", "Alder skal være mellem 0 og 120.")
            return

        # Køn: 1 = mand, 0 = kvinde (matcher træningen)
        sex = 1 if self.sex_checkbox.isChecked() else 0

        # Sygdomme 0/1
        disease_values = {
            key: int(cb.isChecked())
            for key, cb in self.disease_checkboxes.items()}

        # Ekstra features
        age_dummies = self.age_cat_features(age)
        comorb_dummies = self.comorbidity_category(disease_values)
        inter_feats = self.interaction_features(age, sex, disease_values)

        # Saml alle features i én dict
        row = {
            "AGE": age,
            "SEX": sex,
            **disease_values,
            **age_dummies,
            **comorb_dummies,
            **inter_feats,}

        # Konverter til model-input i korrekt rækkefølge
        X = np.array([[row[feat] for feat in self.feature_order]], dtype=float)

        # Rå model-sandsynlighed
        raw_prob = float(self.model.predict_proba(X)[0, 1])

        # Kalibreret sandsynlighed via IsotonicRegression
        prob = float(self.calibrator.predict([raw_prob])[0])

        # Risikoniveau
        level, color = risk_level(prob)

        palette = self.result_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, color)
        self.result_label.setPalette(palette)

        self.result_label.setText(
            f"Risiko: {level}\nSandsynlighed for død: {prob*100:.2f}%\nRisikoprofil skal altid sammenholdes med klinisk vurdering")

# MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CovidRiskApp()
    win.show()
    sys.exit(app.exec())