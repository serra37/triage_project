import os
import numpy as np
import pandas as pd
import joblib
import shap


# Model dosyalarinin yolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# Model ve araclari yukle
model = joblib.load(os.path.join(MODELS_DIR, "risk_classification_model_ohe.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "risk_classification_imputer_ohe.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "risk_classification_features_ohe.pkl"))


# SHAP explainer model uzerinden yeniden olusturulur
# Boylece explainer pkl surum uyumsuzlugu riski azalir.
explainer = shap.TreeExplainer(model)


# High Risk icin klinik olarak dengeli esik
HIGH_RISK_THRESHOLD = 0.40


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
    0: "Dusuk risk tespit edildi. Semptomlari takip edin, kotulesirse doktora basvurun.",
}


DISCLAIMER = (
    "Bu degerlendirme tani degildir. "
    "Tedavi veya ilac onermez. "
    "Acil durumda 112'yi arayin. "
    "Kesin karar icin saglik profesyoneline basvurun."
)


def _normalize_sex(value):
    """
    Kullanici girdisini modeldeki SEX formatina cevirir.
    Modelde: 0=Erkek, 1=Kadin
    """
    if value is None:
        return np.nan

    if isinstance(value, str):
        val = value.strip().lower()
        if val in ["erkek", "male", "e", "m"]:
            return 0
        if val in ["kadin", "kadın", "female", "k", "f"]:
            return 1

    if value in [0, 1]:
        return value

    # NHAMCS orijinal kodlamasi gelirse: 1=Erkek, 2=Kadin
    if value == 2:
        return 1

    return np.nan


def _normalize_ambulance(value):
    """
    Kullanici girdisini ARREMS formatina cevirir.
    Modelde: 1=Ambulansla geldi, 0=Kendi geldi
    """
    if value is None:
        return np.nan

    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, str):
        val = value.strip().lower()
        if val in ["evet", "yes", "true", "ambulans", "1"]:
            return 1
        if val in ["hayir", "hayır", "no", "false", "0"]:
            return 0

    if value in [0, 1]:
        return value

    # NHAMCS orijinal kodlamasi gelirse: 1=Ambulans, 2=Kendi geldi
    if value == 2:
        return 0

    return np.nan


def _safe_float(value):
    """
    Sayisal degeri guvenli sekilde float'a cevirir.
    Bos veya hatali degerlerde NaN doner.
    """
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _build_input_dataframe(patient_data: dict) -> pd.DataFrame:
    """
    Kullanici/hasta verisini modelin bekledigi feature kolonlarina donusturur.
    Eksik alanlar NaN olarak birakilir, imputer median ile doldurur.
    """

    # 1. Tum feature kolonlarini NaN ile baslat
    input_data = {col: np.nan for col in feature_columns}

    # 2. Kullanici girdisini modele uygun alanlara map et
    field_mapping = {
        "age": "AGE",
        "pulse": "PULSE",
        "respr": "RESPR",
        "bpsys": "BPSYS",
        "bpdias": "BPDIAS",
        "popct": "POPCT",
        "tempc": "TEMPC",
        "painscale": "PAINSCALE",
        "totchron": "TOTCHRON",
    }

    for key, model_key in field_mapping.items():
        if key in patient_data and model_key in input_data:
            input_data[model_key] = _safe_float(patient_data.get(key))

    # 3. SEX normalize et
    if "SEX" in input_data:
        input_data["SEX"] = _normalize_sex(patient_data.get("sex"))

    # 4. ARREMS normalize et
    if "ARREMS" in input_data:
        input_data["ARREMS"] = _normalize_ambulance(patient_data.get("arrems"))

    # 5. PAINSCALE_MISSING gostergesi
    if "PAINSCALE_MISSING" in input_data:
        painscale_val = patient_data.get("painscale")
        input_data["PAINSCALE_MISSING"] = 1 if painscale_val is None else 0

    # 6. RFV1_GROUP one-hot encoding
    # Final OHE modelinde kolonlar RFV_100.0, RFV_101.0 gibi isimlendirilmiştir.
    rfv1_group = patient_data.get("rfv1_group")
    if rfv1_group is not None:
        try:
            rfv_col = f"RFV_{float(rfv1_group)}"
            if rfv_col in input_data:
                input_data[rfv_col] = 1
        except (TypeError, ValueError):
            pass

    # 7. DataFrame olustur
    return pd.DataFrame([input_data], columns=feature_columns)


def _get_shap_for_class(df_imputed: pd.DataFrame, risk_class: int) -> np.ndarray:
    """
    SHAP cikti formatlari surume gore degisebilir.
    Bu fonksiyon list, 3D array ve 2D array formatlarini guvenli sekilde ele alir.
    """

    shap_values = explainer.shap_values(df_imputed)
    shap_values_arr = np.array(shap_values)

    # Eski format: [class][sample][feature]
    if isinstance(shap_values, list):
        return np.array(shap_values[risk_class][0])

    # Yeni format: [sample, feature, class]
    if shap_values_arr.ndim == 3:
        return shap_values_arr[0, :, risk_class]

    # Binary veya tek cikti formati: [sample, feature]
    if shap_values_arr.ndim == 2:
        return shap_values_arr[0]

    # Beklenmeyen durumda sistemin patlamasini engelle
    return np.zeros(len(feature_columns))


def _create_shap_explanation(shap_for_class: np.ndarray, top_n: int = 3) -> list:
    """
    En etkili SHAP feature'larini Turkce aciklama olarak dondurur.
    """

    feature_shap = list(zip(feature_columns, shap_for_class))
    feature_shap_sorted = sorted(
        feature_shap,
        key=lambda x: abs(float(x[1])),
        reverse=True
    )

    top_features = feature_shap_sorted[:top_n]

    explanations = []
    for feat, val in top_features:
        feat_name = FEATURE_LABELS_TR.get(feat, feat)

        # One-hot RFV kolonlari icin daha okunabilir ad
        if feat.startswith("RFV_"):
            feat_name = f"Basvuru nedeni grubu ({feat.replace('RFV_', '')})"

        direction = "riski artirdi" if val > 0 else "riski azaltti"
        explanations.append(f"{feat_name} {direction}")

    return explanations


def predict_risk(patient_data: dict) -> dict:
    """
    Hasta verisinden risk tahmini yapar ve SHAP aciklamasi uretir.

    Parametreler:
        patient_data: dict
            Asagidaki alanlari icerebilir:
            age, sex, pulse, respr, bpsys, bpdias, popct,
            tempc, arrems, painscale, totchron, rfv1_group

    Donus degeri:
        dict:
            risk_level       : int (0=Low, 1=Medium, 2=High)
            risk_label       : str
            risk_label_tr    : str
            risk_color       : str
            confidence       : float
            probabilities    : dict
            shap_explanation : list of str
            recommendation   : str
            disclaimer       : str
    """

    # 1. Input hazirla
    df_input = _build_input_dataframe(patient_data)

    # 2. Imputation
    df_imputed = pd.DataFrame(
        imputer.transform(df_input),
        columns=feature_columns
    )

    # 3. Risk olasiliklari
    risk_proba = model.predict_proba(df_imputed)[0]

    # 4. Threshold destekli risk karari
    # High Risk olasiligi 0.40 ve uzerindeyse High kabul edilir.
    if risk_proba[2] >= HIGH_RISK_THRESHOLD:
        risk_class = 2
    else:
        risk_class = int(np.argmax(risk_proba))

    confidence = float(risk_proba[risk_class])

    # 5. SHAP aciklamasi
    shap_for_class = _get_shap_for_class(df_imputed, risk_class)
    shap_explanation = _create_shap_explanation(shap_for_class, top_n=3)

    return {
        "risk_level": risk_class,
        "risk_label": RISK_MAPPING[risk_class]["label"],
        "risk_label_tr": RISK_MAPPING[risk_class]["tr"],
        "risk_color": RISK_MAPPING[risk_class]["color"],
        "confidence": round(confidence, 3),
        "probabilities": {
            "Low": round(float(risk_proba[0]), 3),
            "Medium": round(float(risk_proba[1]), 3),
            "High": round(float(risk_proba[2]), 3),
        },
        "threshold": HIGH_RISK_THRESHOLD,
        "shap_explanation": shap_explanation,
        "recommendation": RECOMMENDATIONS[risk_class],
        "disclaimer": DISCLAIMER,
    }


def check_minimum_data(patient_data: dict) -> tuple:
    """
    Modelin calisabilmesi icin minimum veri var mi kontrol eder.

    Donus degeri:
        (yeterli_mi: bool, eksik_alanlar: list)
    """

    minimum_required = ["age"]
    missing = [
        field for field in minimum_required
        if field not in patient_data or patient_data.get(field) is None
    ]

    return len(missing) == 0, missing


def format_risk_response(result: dict) -> str:
    """
    predict_risk ciktisini kullaniciya gosterilecek Turkce metne donusturur.
    """

    probabilities = result.get("probabilities", {})

    lines = [
        f"Risk Seviyesi: {result['risk_label_tr']}",
        f"Guven Skoru: %{int(result['confidence'] * 100)}",
        "",
        "Sinif olasiliklari:",
        f"  - Dusuk Risk: %{int(probabilities.get('Low', 0) * 100)}",
        f"  - Orta Risk: %{int(probabilities.get('Medium', 0) * 100)}",
        f"  - Yuksek Risk: %{int(probabilities.get('High', 0) * 100)}",
        "",
        "Bu kararda etkili olan faktorler:",
    ]

    for explanation in result.get("shap_explanation", []):
        lines.append(f"  - {explanation}")

    lines.append("")
    lines.append(result["recommendation"])
    lines.append("")
    lines.append(result["disclaimer"])

    return "\n".join(lines)