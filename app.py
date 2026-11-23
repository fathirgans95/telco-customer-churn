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
    page_title="Telco Churn AI | Pink Edition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS: Dominan Pink + Efek Glow, Glassmorphism & Animasi Hover/Pulse
st.markdown("""
<style>
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 18px rgba(236, 72, 153, 0.25), inset 0 0 12px rgba(244, 114, 182, 0.08); }
        50% { box-shadow: 0 0 32px rgba(236, 72, 153, 0.48), inset 0 0 18px rgba(244, 114, 182, 0.16); }
        100% { box-shadow: 0 0 18px rgba(236, 72, 153, 0.25), inset 0 0 12px rgba(244, 114, 182, 0.08); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #3b0a2a 0%, #1d0718 45%, #0f050d 100%);
        color: #fdf2f8;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    .top-banner {
        background: linear-gradient(135deg, rgba(80, 14, 55, 0.85), rgba(40, 10, 32, 0.92));
        border: 1px solid rgba(244, 114, 182, 0.45);
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        animation: pulseGlow 4s infinite ease-in-out;
    }

    .badge-pill {
        display: inline-block;
        background: linear-gradient(90deg, rgba(236, 72, 153, 0.28), rgba(244, 63, 94, 0.28));
        color: #f9a8d4;
        border: 1px solid rgba(244, 114, 182, 0.6);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        box-shadow: 0 0 12px rgba(236, 72, 153, 0.35);
    }

    .panel-card {
        background: rgba(38, 10, 30, 0.78);
        border: 1px solid rgba(244, 114, 182, 0.28);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .panel-card:hover {
        border-color: rgba(244, 114, 182, 0.65);
        box-shadow: 0 10px 34px rgba(236, 72, 153, 0.22);
    }

    .metric-box {
        background: linear-gradient(145deg, rgba(65, 16, 48, 0.75), rgba(35, 9, 27, 0.85));
        border: 1px solid rgba(244, 114, 182, 0.35);
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 12px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .metric-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.35);
    }

    .metric-label {
        font-size: 12px;
        color: #f9a8d4;
        margin-bottom: 4px;
        font-weight: 500;
    }

    .metric-val {
        font-size: 28px;
        font-weight: 800;
        text-shadow: 0 0 14px rgba(244, 114, 182, 0.45);
    }

    .status-churn {
        background: linear-gradient(90deg, rgba(225, 29, 72, 0.25), rgba(190, 24, 93, 0.25));
        border: 1px solid rgba(251, 113, 133, 0.7);
        color: #fecdd3;
        border-radius: 14px;
        padding: 15px 18px;
        font-weight: 600;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(225, 29, 72, 0.3);
    }

    .status-stay {
        background: linear-gradient(90deg, rgba(236, 72, 153, 0.22), rgba(217, 70, 239, 0.22));
        border: 1px solid rgba(244, 114, 182, 0.7);
        color: #fbcfe8;
        border-radius: 14px;
        padding: 15px 18px;
        font-weight: 600;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.3);
    }

    /* Styling Tab Navigasi Pink */
    button[data-baseweb="tab"] {
        color: #f9a8d4 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #ec4899 !important;
        text-shadow: 0 0 10px rgba(236, 72, 153, 0.8);
    }

    /* Tombol Utama Pink Gradient + Efek Hover Terangkat */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(270deg, #ec4899, #f43f5e, #d946ef, #ec4899);
        background-size: 300% 300%;
        animation: gradientShift 5s ease infinite;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        padding: 11px 20px;
        font-weight: 700;
        letter-spacing: 0.3px;
        width: 100%;
        box-shadow: 0 4px 18px rgba(236, 72, 153, 0.45);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 26px rgba(236, 72, 153, 0.75);
    }
</style>
""", unsafe_allow_html=True)

# Tema Grafik Matplotlib Bernuansa Pink Gelap
plt.rcParams.update({
    "figure.facecolor": "#1d0718",
    "axes.facecolor": "#1d0718",
    "axes.edgecolor": "#831843",
    "axes.labelcolor": "#fbcfe8",
    "xtick.color": "#f9a8d4",
    "ytick.color": "#f9a8d4",
    "text.color": "#fdf2f8",
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

# Header Banner Pink
st.markdown("""
<div class="top-banner">
    <div class="badge-pill">🌸 TELCO CHURN AI • PINK GLOW EDITION (IDR)</div>
    <h2 style="margin:0; font-size:28px; font-weight:800; color:#fdf2f8;">💖 Prediksi Pelanggan Berhenti (Telco Churn)</h2>
    <p style="margin:6px 0 0 0; color:#f9a8d4; font-size:14px;">
        Simulator prediksi churn pelanggan telekomunikasi berbasis Machine Learning dengan input &amp; tampilan Rupiah (1 USD = Rp16.000).
    </p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_eval, tab_method = st.tabs([
    "💖 Prediction Lab",
    "📊 Evaluasi Model",
    "✨ Metodologi",
])

# ================= TAB 1: PREDICTION LAB =================
with tab_pred:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Parameter Pelanggan")
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

        run_btn = st.button("✨ Jalankan Prediksi Churn", type="primary")
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
        st.markdown(f"### 🔮 Hasil Prediksi ({model_choice})")

        if pred == 1:
            st.markdown(
                f'<div class="status-churn">🚨 Berisiko Churn (Yes) — Estimasi probabilitas berhenti sebesar {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-stay">💖 Cenderung Bertahan (No) — Estimasi probabilitas churn hanya {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Bertahan (No)</div><div class="metric-val" style="color:#f472b6;">{prob_stay:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Churn (Yes)</div><div class="metric-val" style="color:#fb7185;">{prob_churn:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        fig_pred, ax_pred = plt.subplots(figsize=(6.2, 3.0))
        labels = ["Bertahan (No)", "Berisiko Churn (Yes)"]
        values = [prob_stay, prob_churn]
        colors = ["#f472b6", "#e11d48"]
        bars = ax_pred.barh(labels, values, color=colors, height=0.48, edgecolor="#fbcfe8", linewidth=0.8)
        ax_pred.set_xlim(0, 112)
        ax_pred.set_xlabel("Probabilitas (%)")
        ax_pred.set_title("Distribusi Probabilitas Pelanggan", fontsize=11, fontweight="bold", color="#fdf2f8")
        ax_pred.grid(axis="x", linestyle="--", alpha=0.25, color="#f472b6")
        for bar, val in zip(bars, values):
            ax_pred.text(
                val + 1.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontweight="bold",
                color="#fdf2f8",
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
    st.markdown("### 📊 Ringkasan Performa Model pada Data Uji")

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
        cmap="RdPu",
        ax=axes[0],
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
    )
    axes[0].set_title("Confusion Matrix — Logistic Regression", color="#fdf2f8")

    feat_imp = (
        pd.Series(rf_model.feature_importances_, index=model_columns)
        .sort_values(ascending=False)
        .head(10)
    )
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[1], palette="flare")
    axes[1].set_title("Top 10 Faktor Paling Berpengaruh terhadap Churn", color="#fdf2f8")

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: METODOLOGI =================
with tab_method:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### ✨ Alur Pemrosesan & Konversi Mata Uang")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Preprocessing Data**")
        st.caption("Konversi `TotalCharges` ke numerik, hapus baris kosong, encoding fitur kategorikal, dan standarisasi fitur numerik (`StandardScaler`).")
    with col_b:
        st.markdown("**2. Konversi Rupiah (`Rp`)**")
        st.caption("Input tagihan bulanan dan total tagihan dalam Rupiah dibagi `16.000` (`1 USD = Rp16.000`) sebelum masuk ke `scaler.transform()` agar skala tetap selaras dengan data latih.")
    with col_c:
        st.markdown("**3. Klasifikasi Churn**")
        st.caption("Melatih model *Logistic Regression* dan *Random Forest* dengan pembagian data *train-test split* stratified (80:20).")
    st.markdown("</div>", unsafe_allow_html=True)
