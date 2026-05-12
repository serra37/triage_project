import os
import numpy as np
import pandas as pd
import joblib
import shap

# Model dosyalarinin yolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "risk_classification_model_ohe.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "risk_classification_imputer_ohe.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "risk_classification_features_ohe.pkl"))

# SHAP explainer — model uzerinden yeniden olustur (surum uyumsuzlugunu onler)
explainer = shap.TreeExplainer(model)

# Risk etiketi mapping
RISK_MAPPING = {
    0: {"label": "Low", "tr": "Dusuk Risk", "color": "green"},
    1: {"label": "Medium", "tr": "Orta Risk", "color": "orange"},
    2: {"label": "High", "tr": "Yuksek Risk", "color": "red"},
}

# Turkce feature isimleri
FEATURE_LABELS_TR = {
    "AGE": "Yas",
    "SEX": "Cinsiyet",
    "PULSE": "Nabiz",
    "RESPR": "Solunum hizi",
    "BPSYS": "Sistolik tansiyon",
    "BPDIAS": "Diastolik tansiyon",
    "POPCT": "Oksijen saturasyonu",
    "TEMPC": "Vucut sicakligi",
    "ARREMS": "Ambulansla gelis",
    "PAINSCALE": "Agri siddeti",
    "PAINSCALE_MISSING": "Agri bilgisi eksik",
    "TOTCHRON": "Kronik hastalik sayisi",
}

# Yonlendirme mesajlari
RECOMMENDATIONS = {
    2: "Yuksek risk tespit edildi. En kisa surede acil servise basvurmaniz onerilir.",
    1: "Orta risk tespit edildi. Bugun icinde bir doktora basvurmaniz onerilir.",
    0: "Dusuk risk tespit edildi. Semptomlari takip edin, kotulesirsre doktora basvurun.",
}

DISCLAIMER = (
    "Bu degerlendirme tani degildir. "
    "Kesin karar icin saglik profesyoneline basvurun."
)


def predict_risk(patient_data: dict) -> dict:
    """
    Hasta verisinden risk tahmini yap ve SHAP aciklamasi uret.

    Parametreler:
        patient_data: dict
            Asagidaki alanlari iceren sozluk:
            age, sex, pulse, respr, bpsys, bpdias, popct,
            tempc, arrems, painscale, totchron, rfv1_group

    Donus degeri:
        dict:
            risk_level       : int (0=Low, 1=Medium, 2=High)
            risk_label       : str
            risk_label_tr    : str
            confidence       : float
            shap_explanation : list of str
            recommendation   : str
            disclaimer       : str
    """

    # 1. Tum feature kolonlarini NaN ile baslat
    input_data = {col: np.nan for col in feature_columns}

    # 2. Kullanici girdisini modele uygun alanlara map et
    field_mapping = {
        "age": "AGE",
        "sex": "SEX",
        "pulse": "PULSE",
        "respr": "RESPR",
        "bpsys": "BPSYS",
        "bpdias": "BPDIAS",
        "popct": "POPCT",
        "tempc": "TEMPC",
        "arrems": "ARREMS",
        "painscale": "PAINSCALE",
        "totchron": "TOTCHRON",
    }

    for key, value in patient_data.items():
        model_key = field_mapping.get(key.lower())
        if model_key and model_key in input_data:
            input_data[model_key] = value

    # 3. PAINSCALE_MISSING gostergesi
    if "PAINSCALE_MISSING" in input_data:
        painscale_val = patient_data.get("painscale")
        input_data["PAINSCALE_MISSING"] = 1 if painscale_val is None else 0

    # 4. RFV1_GROUP one-hot encoding
    rfv1_group = patient_data.get("rfv1_group")
    if rfv1_group is not None:
        rfv_col = f"RFV1_GROUP_{float(rfv1_group)}"
        if rfv_col in input_data:
            input_data[rfv_col] = 1

    # 5. DataFrame olustur
    df_input = pd.DataFrame([input_data], columns=feature_columns)

    # 6. Imputation — sklearn surum farki nedeniyle manuel median imputation
    df_imputed = df_input.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isna().any():
            df_imputed[col] = df_imputed[col].fillna(0)

    # 7. Risk tahmini
    risk_class = int(model.predict(df_imputed)[0])
    risk_proba = model.predict_proba(df_imputed)[0]
    confidence = float(risk_proba[risk_class])

    # 8. SHAP aciklamasi
    shap_values = explainer.shap_values(df_imputed)
    
    # SHAP output formatini kontrol et
    if isinstance(shap_values, list):
        # Eski format: liste [sinif0, sinif1, sinif2]
        shap_for_class = shap_values[risk_class][0]
    else:
        # Yeni format: (n_samples, n_features, n_classes)
        shap_for_class = shap_values[0, :, risk_class]

    # Top 3 etkili feature
    feature_shap = list(zip(feature_columns, shap_for_class))
    feature_shap_sorted = sorted(
        feature_shap, key=lambda x: abs(x[1]), reverse=True
    )
    top_features = feature_shap_sorted[:3]

    # Turkce aciklama uret
    shap_explanation = []
    for feat, val in top_features:
        feat_name = FEATURE_LABELS_TR.get(feat, feat)
        direction = "riski artirdi" if val > 0 else "riski azaltti"
        shap_explanation.append(f"{feat_name} {direction}")

    return {
        "risk_level": risk_class,
        "risk_label": RISK_MAPPING[risk_class]["label"],
        "risk_label_tr": RISK_MAPPING[risk_class]["tr"],
        "confidence": round(confidence, 3),
        "shap_explanation": shap_explanation,
        "recommendation": RECOMMENDATIONS[risk_class],
        "disclaimer": DISCLAIMER,
    }


def check_minimum_data(patient_data: dict) -> tuple:
    """
    Modelin calisabilmesi icin minimum veri var mi kontrol et.

    Donus degeri:
        (yeterli_mi: bool, eksik_alanlar: list)
    """
    minimum_required = ["age"]
    missing = [f for f in minimum_required if f not in patient_data]
    return len(missing) == 0, missing


def format_risk_response(result: dict) -> str:
    """
    predict_risk ciktisini kullaniciya gosterilecek
    Turkce metin formatina donustur.
    """
    lines = [
        f"Risk Seviyesi: {result['risk_label_tr']}",
        f"Guven Skoru: %{int(result['confidence'] * 100)}",
        "",
        "Bu kararda etkili olan faktorler:",
    ]
    for explanation in result["shap_explanation"]:
        lines.append(f"  - {explanation}")

    lines.append("")
    lines.append(result["recommendation"])
    lines.append("")
    lines.append(result["disclaimer"])

    return "\n".join(lines)
