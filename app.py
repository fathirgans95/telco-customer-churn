import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

KURS_IDR = 16000  # 1 USD = Rp16.000

st.set_page_config(
    page_title="Telco Churn AI | Soft Pink Edition",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS: Pink Muda Pastel / Sakura Girly + Efek Shimmer, Float & Soft Glow
st.markdown("""
<style>
    @keyframes floatSoft {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
        100% { transform: translateY(0px); }
    }

    @keyframes shimmerMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes softPulsePink {
        0% { box-shadow: 0 10px 28px rgba(244, 114, 182, 0.22); }
        50% { box-shadow: 0 14px 36px rgba(244, 114, 182, 0.38); }
        100% { box-shadow: 0 10px 28px rgba(244, 114, 182, 0.22); }
    }

    .stApp {
        background: linear-gradient(160deg, #fff5f8 0%, #ffe4ee 48%, #ffd6e7 100%);
        color: #5c1d3b;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    .top-banner {
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(255, 228, 238, 0.92), rgba(255, 240, 246, 0.92));
        border: 2px solid #f9a8d4;
        border-radius: 26px;
        padding: 22px 28px;
        margin-bottom: 20px;
        animation: softPulsePink 4.5s infinite ease-in-out;
    }

    .badge-pill {
        display: inline-block;
        background: linear-gradient(90deg, #fbcfe8, #fde2e4);
        color: #9d174d;
        border: 1px solid #f472b6;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
        animation: floatSoft 3.2s infinite ease-in-out;
    }

    .panel-card {
        background: rgba(255, 250, 252, 0.9);
        border: 1.5px solid #f9a8d4;
        border-radius: 24px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 26px rgba(236, 72, 153, 0.12);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }

    .panel-card:hover {
        transform: translateY(-3px);
        border-color: #f472b6;
        box-shadow: 0 14px 32px rgba(236, 72, 153, 0.22);
    }

    .metric-box {
        background: linear-gradient(145deg, #fff0f6, #ffe4ef);
        border: 1.5px solid #f9a8d4;
        border-radius: 20px;
        padding: 15px 18px;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(244, 114, 182, 0.14);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .metric-box:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 10px 22px rgba(236, 72, 153, 0.24);
    }

    .metric-label {
        font-size: 12px;
        color: #9d174d;
        margin-bottom: 4px;
        font-weight: 700;
    }

    .metric-val {
        font-size: 28px;
        font-weight: 800;
    }

    .status-churn {
        background: linear-gradient(90deg, #ffe4e6, #fecdd3);
        border: 1.5px solid #fb7185;
        color: #9f1239;
        border-radius: 18px;
        padding: 15px 18px;
        font-weight: 700;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(251, 113, 133, 0.2);
    }

    .status-stay {
        background: linear-gradient(90deg, #fce7f3, #fbcfe8);
        border: 1.5px solid #f472b6;
        color: #831843;
        border-radius: 18px;
        padding: 15px 18px;
        font-weight: 700;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(244, 114, 182, 0.22);
    }

    /* Tab Navigasi Pastel Pink */
    button[data-baseweb="tab"] {
        color: #9d174d !important;
        font-weight: 700 !important;
        border-radius: 14px 14px 0 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #db2777 !important;
        background: rgba(255, 255, 255, 0.75) !important;
        border-bottom-color: #f472b6 !important;
    }

    /* Tombol Utama Pastel Pink Shimmer */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #f472b6, #fb7185, #f9a8d4, #f472b6);
        background-size: 300% 300%;
        animation: shimmerMove 4.5s ease infinite;
        color: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 999px;
        padding: 11px 22px;
        font-weight: 800;
        letter-spacing: 0.4px;
        width: 100%;
        box-shadow: 0 8px 22px rgba(244, 114, 182, 0.45);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.015);
        box-shadow: 0 12px 28px rgba(236, 72, 153, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Tema Grafik Matplotlib Pink Muda Lembut
plt.rcParams.update({
    "figure.facecolor": "#fff7fa",
    "axes.facecolor": "#fff7fa",
    "axes.edgecolor": "#f9a8d4",
    "axes.labelcolor": "#831843",
    "xtick.color": "#9d174d",
    "ytick.color": "#9d174d",
    "text.color": "#5c1d3b",
})


@st.cache_resource
def load_assets():
    url = "https://raw.githubusercontent.com/fathirgans95/telco-customer-churn/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X_encoded = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model, scaler, list(X_train.columns), X_test, y_test


log_model, rf_model, scaler, model_columns, X_test, y_test = load_assets()

# Header Banner Pink Muda
st.markdown("""
<div class="top-banner">
    <div class="badge-pill">🎀 TELCO CHURN AI • SAKURA PASTEL EDITION (IDR)</div>
    <h2 style="margin:0; font-size:28px; font-weight:800; color:#831843;">🌸 Prediksi Pelanggan Berhenti (Telco Churn)</h2>
    <p style="margin:6px 0 0 0; color:#9d174d; font-size:14px;">
        Simulator prediksi churn pelanggan telekomunikasi berbasis Machine Learning dengan input &amp; tampilan Rupiah (1 USD = Rp16.000).
    </p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_eval, tab_method = st.tabs([
    "🌷 Prediction Lab",
    "💗 Evaluasi Model",
    "🎀 Metodologi",
])

# ================= TAB 1: PREDICTION LAB =================
with tab_pred:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### 🌷 Parameter Pelanggan")
        st.caption("Atur profil layanan dan tagihan dalam Rupiah lalu jalankan prediksi.")

        model_choice = st.radio(
            "Model Prediksi",
            ["Logistic Regression", "Random Forest"],
            horizontal=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with c2:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            payment_method = st.selectbox("Metode Pembayaran", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])

        tenure = st.slider("Lama Berlangganan / Tenure (Bulan)", 1, 72, 12)

        t1, t2 = st.columns(2)
        with t1:
            monthly_charges_idr = st.number_input(
                "Tagihan Bulanan (Rp)",
                min_value=0,
                max_value=5000000,
                value=1120000,
                step=50000,
            )
        with t2:
            total_charges_idr = st.number_input(
                "Total Tagihan (Rp)",
                min_value=0,
                max_value=200000000,
                value=int(tenure * monthly_charges_idr),
                step=100000,
            )

        st.caption(
            f"Tagihan: **Rp {monthly_charges_idr:,.0f}/bln** • Total: **Rp {total_charges_idr:,.0f}**".replace(",", ".")
        )

        run_btn = st.button("🎀 Jalankan Prediksi Churn", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    monthly_usd = monthly_charges_idr / KURS_IDR
    total_usd = total_charges_idr / KURS_IDR

    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_usd,
        "TotalCharges": total_usd,
    }
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    active_model = log_model if model_choice == "Logistic Regression" else rf_model
    pred = active_model.predict(input_encoded)[0]
    proba = active_model.predict_proba(input_encoded)[0]
    prob_stay, prob_churn = proba[0] * 100, proba[1] * 100

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f"### 💕 Hasil Prediksi ({model_choice})")

        if pred == 1:
            st.markdown(
                f'<div class="status-churn">🍓 Berisiko Churn (Yes) — Estimasi probabilitas berhenti sebesar {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-stay">🌸 Cenderung Bertahan (No) — Estimasi probabilitas churn hanya {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Bertahan (No)</div><div class="metric-val" style="color:#db2777;">{prob_stay:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Churn (Yes)</div><div class="metric-val" style="color:#f43f5e;">{prob_churn:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        fig_pred, ax_pred = plt.subplots(figsize=(6.2, 3.0))
        labels = ["Bertahan (No)", "Berisiko Churn (Yes)"]
        values = [prob_stay, prob_churn]
        colors = ["#f9a8d4", "#fb7185"]
        bars = ax_pred.barh(labels, values, color=colors, height=0.48, edgecolor="#ec4899", linewidth=1.2)
        ax_pred.set_xlim(0, 112)
        ax_pred.set_xlabel("Probabilitas (%)")
        ax_pred.set_title("Distribusi Probabilitas Pelanggan", fontsize=11, fontweight="bold", color="#831843")
        ax_pred.grid(axis="x", linestyle="--", alpha=0.45, color="#f472b6")
        for bar, val in zip(bars, values):
            ax_pred.text(
                val + 1.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontweight="bold",
                color="#831843",
            )
        plt.tight_layout()
        st.pyplot(fig_pred)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 2: EVALUASI MODEL =================
with tab_eval:
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### 💗 Ringkasan Performa Model pada Data Uji")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LogReg Accuracy", f"{accuracy_score(y_test, y_pred_log)*100:.2f}%")
    m2.metric("LogReg ROC-AUC", f"{roc_auc_score(y_test, y_prob_log):.4f}")
    m3.metric("RF Accuracy", f"{accuracy_score(y_test, y_pred_rf)*100:.2f}%")
    m4.metric("RF ROC-AUC", f"{roc_auc_score(y_test, y_prob_rf):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    cm = confusion_matrix(y_test, y_pred_log)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="pink",
        ax=axes[0],
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
    )
    axes[0].set_title("Confusion Matrix — Logistic Regression", color="#831843")

    feat_imp = (
        pd.Series(rf_model.feature_importances_, index=model_columns)
        .sort_values(ascending=False)
        .head(10)
    )
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[1], palette="pastel")
    axes[1].set_title("Top 10 Faktor Paling Berpengaruh terhadap Churn", color="#831843")

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: METODOLOGI =================
with tab_method:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### 🎀 Alur Pemrosesan & Konversi Mata Uang")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Preprocessing Data**")
        st.caption("
