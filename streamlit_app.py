from pathlib import Path
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_5feature_lightgbm_model.txt"
CONFIG_PATH = APP_DIR / "online_calculator_model_config.csv"
FEATURE_ORDER_PATH = APP_DIR / "clinical_feature_order.csv"

st.set_page_config(
    page_title="Neonatal FI Risk Calculator",
    page_icon="🍼",
    layout="wide",
)

@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(MODEL_PATH))

@st.cache_data
def load_config():
    cfg = pd.read_csv(CONFIG_PATH)
    return dict(zip(cfg["Item"].astype(str), cfg["Value"].astype(str)))

@st.cache_data
def load_feature_order():
    df = pd.read_csv(FEATURE_ORDER_PATH)
    return df.sort_values("Order")["Clinical_Feature"].astype(str).tolist()

booster = load_model()
config = load_config()
feature_order = load_feature_order()

EXPECTED_FEATURES = [
    "feeddelay",
    "bloodtransfusion",
    "gestationalgeweek",
    "tbil",
    "birthweight",
]

if feature_order != EXPECTED_FEATURES:
    st.error("Feature-order check failed. The deployed model files are inconsistent.")
    st.stop()

if booster.feature_name() != EXPECTED_FEATURES:
    st.error("Model feature-name check failed. The deployed model files are inconsistent.")
    st.stop()

threshold = float(config["Classification_threshold"])
internal_auc = float(config["AUC_recalculated_from_saved_model"])

# Development-data ranges stored inside the exported LightGBM model.
DEVELOPMENT_RANGE = {
    "gestationalgeweek": (26.0, 41.0),
    "tbil": (5.7, 309.4),
    "birthweight": (890.0, 3950.0),
}

DISPLAY_NAME = {
    "feeddelay": "Delayed feeding",
    "bloodtransfusion": "Blood transfusion",
    "gestationalgeweek": "Gestational age",
    "tbil": "Total bilirubin",
    "birthweight": "Birth weight",
}

st.title("Neonatal Feeding Intolerance Risk Calculator")
st.caption(
    "Interpretable 5-feature LightGBM model for predicting feeding intolerance (FI)"
)

st.info(
    "This calculator reproduces the fixed final LightGBM model used in the study. "
    "It does not re-train, re-tune, or recalibrate the model."
)
st.caption(
    "Intended users: NICU clinicians and researchers familiar with neonatal care. "
    "All five predictor fields must be explicitly completed before a prediction can be generated."
)

left, right = st.columns([1.05, 1.25], gap="large")

with left:
    st.subheader("Patient information")
    st.caption("* All five fields are required. No default patient values are used.")

    feeding_delay_label = st.selectbox(
        "Delayed feeding (>24 h) *",
        options=["No (≤24 h)", "Yes (>24 h)"],
        index=None,
        placeholder="Select an option",
        help=(
            "Defined by time to initiation of enteral feeding after birth: "
            "No = ≤24 h; Yes = >24 h."
        ),
    )
    if feeding_delay_label is None:
        feeddelay = None
    else:
        feeddelay = 1 if feeding_delay_label == "Yes (>24 h)" else 0

    transfusion_label = st.selectbox(
        "Blood transfusion *",
        options=["No", "Yes"],
        index=None,
        placeholder="Select an option",
        help=(
            "No = 0; Yes = 1. Use the same definition and time window as in the study dataset "
            "(within the first 14 days after birth)."
        ),
    )
    if transfusion_label is None:
        bloodtransfusion = None
    else:
        bloodtransfusion = 1 if transfusion_label == "Yes" else 0

    gestationalgeweek = st.number_input(
        "Gestational age (weeks) *",
        value=None,
        step=0.1,
        format="%.1f",
        placeholder="Enter gestational age",
        help="Development-data range in the fitted model: 26–41 weeks.",
    )

    tbil = st.number_input(
        "Total bilirubin (μmol/L) *",
        value=None,
        step=0.1,
        format="%.2f",
        placeholder="Enter total bilirubin",
        help=(
            "Enter total bilirubin in μmol/L. "
            "Development-data range in the fitted model: 5.7–309.4 μmol/L."
        ),
    )

    birthweight = st.number_input(
        "Birth weight (g) *",
        value=None,
        step=10.0,
        format="%.0f",
        placeholder="Enter birth weight",
        help="Development-data range in the fitted model: 890–3950 g.",
    )

    required_inputs = {
        "Delayed feeding": feeddelay,
        "Blood transfusion": bloodtransfusion,
        "Gestational age": gestationalgeweek,
        "Total bilirubin": tbil,
        "Birth weight": birthweight,
    }
    missing_fields = [
        name for name, value in required_inputs.items()
        if value is None
    ]
    all_fields_complete = len(missing_fields) == 0

    outside = []
    for key, value in {
        "gestationalgeweek": gestationalgeweek,
        "tbil": tbil,
        "birthweight": birthweight,
    }.items():
        if value is None:
            continue
        low, high = DEVELOPMENT_RANGE[key]
        if value < low or value > high:
            outside.append(
                f"{DISPLAY_NAME[key]} ({value:g}; development range {low:g}–{high:g})"
            )

    if outside:
        st.warning(
            "One or more values are outside the range observed in the model-development data: "
            + "; ".join(outside)
            + ". A prediction can still be generated, but it represents extrapolation beyond "
              "the observed development-data range and should be interpreted cautiously."
        )

    if missing_fields:
        st.info(
            "Complete all five required fields before calculation. Missing: "
            + ", ".join(missing_fields)
            + "."
        )

    calculate = st.button(
        "Calculate FI probability",
        type="primary",
        use_container_width=True,
        disabled=not all_fields_complete,
    )

with right:
    st.subheader("Prediction")

    if calculate:
        x = pd.DataFrame(
            [[
                feeddelay,
                bloodtransfusion,
                float(gestationalgeweek),
                float(tbil),
                float(birthweight),
            ]],
            columns=EXPECTED_FEATURES,
        )

        probability = float(booster.predict(x)[0])
        percent = probability * 100.0
        above = probability >= threshold

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Predicted probability of FI",
                f"{percent:.1f}%",
            )
        with c2:
            st.metric(
                "Prespecified threshold",
                f"{threshold:.3f}",
            )

        st.progress(min(max(probability, 0.0), 1.0))

        if above:
            st.warning(
                "The predicted probability is at or above the prespecified model threshold "
                f"({threshold:.3f})."
            )
        else:
            st.success(
                "The predicted probability is below the prespecified model threshold "
                f"({threshold:.3f})."
            )

        st.subheader("Individual SHAP explanation")

        contrib = booster.predict(x, pred_contrib=True)[0]
        feature_contrib = np.asarray(contrib[:-1], dtype=float)
        base_value = float(contrib[-1])

        shap_df = pd.DataFrame({
            "Feature": [DISPLAY_NAME[f] for f in EXPECTED_FEATURES],
            "Input value": [
                "Yes (>24 h)" if feeddelay == 1 else "No (≤24 h)",
                "Yes" if bloodtransfusion == 1 else "No",
                f"{gestationalgeweek:g}",
                f"{tbil:g}",
                f"{birthweight:g}",
            ],
            "SHAP contribution (raw score)": feature_contrib,
        })
        shap_df["Direction"] = np.where(
            shap_df["SHAP contribution (raw score)"] > 0,
            "Toward higher FI model output",
            np.where(
                shap_df["SHAP contribution (raw score)"] < 0,
                "Toward lower FI model output",
                "No change",
            ),
        )

        chart_df = (
            shap_df
            .set_index("Feature")[["SHAP contribution (raw score)"]]
            .sort_values("SHAP contribution (raw score)")
        )
        st.bar_chart(chart_df, horizontal=True, height=300)

        st.dataframe(
            shap_df.style.format({
                "SHAP contribution (raw score)": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

        raw_score = base_value + float(feature_contrib.sum())
        reconstructed_probability = 1.0 / (1.0 + math.exp(-raw_score))

        st.caption(
            "Positive SHAP contributions increase the fitted model output for FI; "
            "negative contributions decrease it. SHAP values describe contributions "
            "within the fitted model and should not be interpreted as causal effects."
        )

        with st.expander("Technical check"):
            st.write(f"Base value (raw score): {base_value:.6f}")
            st.write(f"Sum of feature contributions: {feature_contrib.sum():.6f}")
            st.write(f"Reconstructed probability: {reconstructed_probability:.10f}")
            st.write(f"Direct model probability: {probability:.10f}")

    else:
        st.write(
            "Enter the five predictor values and select **Calculate FI probability**."
        )

st.divider()

with st.expander("Model information"):
    st.write(
        f"**Model:** 5-feature LightGBM  \n"
        f"**Predictors:** delayed feeding, blood transfusion, gestational age, "
        f"total bilirubin, and birth weight  \n"
        f"**Fixed decision threshold:** {threshold:.3f}  \n"
        f"**Internal test-set AUC:** {internal_auc:.3f}"
    )
    st.write(
        "The web calculator uses the exported fixed model. "
        "No feature selection, hyperparameter tuning, threshold selection, "
        "or recalibration is performed when a user enters a patient."
    )

st.caption(
    "Research-use tool intended for NICU clinicians and researchers familiar with neonatal care. "
    "The output is intended to support model evaluation and research reporting and should not "
    "replace clinical judgment or be used as a stand-alone treatment or nursing decision rule."
)
