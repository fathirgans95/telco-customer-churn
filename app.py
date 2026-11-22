import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score

KURS_IDR = 16000  # 1 USD = Rp16.000

st.set_page_config(
    page_title="Telco Churn AI | Prediksi Pelanggan",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Gaya visual gelap modern menyerupai dashboard Gradio di video
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #161b2e 0%, #0b0f19 55%, #070a12 100%);
        color: #f1f5f9;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    .top-banner {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 18px;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .panel-card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
    }
    .status-churn {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #fca5a5;
        border-radius: 12px;
        padding: 14px 16px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .status-stay {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #6ee7b7;
        border-radius: 12px;
        padding: 14px 16px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #4338ca, #4f46e5);
    }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#f1f5f9",
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

# Header atas
st.markdown("""
<div class="top-banner">
    <div class="badge-pill">TELCO CHURN AI • RUPIAH (IDR)</div>
    <h2 style="margin:0; font-size:28px; font-weight:700;">📡 Prediksi Pelanggan Berhenti (Telco Churn)</h2>
    <p style="margin:6px 0 0 0; color:#94a3b8; font-size:14px;">
        Simulator prediksi churn pelanggan telekomunikasi berbasis Machine Learning dengan input &amp; tampilan Rupiah (1 USD = Rp16.000).
    </p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_eval, tab_method = st.tabs([
    "🎛️ Prediction Lab",
    "📊 Evaluasi Model",
    "🧭 Metodologi",
])

# ================= TAB 1: PREDICTION LAB (KIRI INPUT, KANAN HASIL) =================
with tab_pred:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Parameter Pelanggan")
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

        run_btn = st.button("Jalankan Prediksi Churn", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Hitung prediksi (default langsung tampil agar halaman langsung hidup seperti dashboard)
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
        st.markdown(f"### Hasil Prediksi ({model_choice})")

        if pred == 1:
            st.markdown(
                f'<div class="status-churn">⚠️ Berisiko Churn (Yes) — Estimasi probabilitas berhenti sebesar {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-stay">✅ Cenderung Bertahan (No) — Estimasi probabilitas churn hanya {prob_churn:.1f}%.</div>',
                unsafe_allow_html=True,
            )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Bertahan (No)</div><div class="metric-val" style="color:#34d399;">{prob_stay:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Peluang Churn (Yes)</div><div class="metric-val" style="color:#f87171;">{prob_churn:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        fig_pred, ax_pred = plt.subplots(figsize=(6.2, 3.0))
        labels = ["Bertahan (No)", "Berisiko Churn (Yes)"]
        values = [prob_stay, prob_churn]
        colors = ["#10b981", "#ef4444"]
        bars = ax_pred.barh(labels, values, color=colors, height=0.48)
        ax_pred.set_xlim(0, 112)
        ax_pred.set_xlabel("Probabilitas (%)")
        ax_pred.set_title("Distribusi Probabilitas Pelanggan", fontsize=11, fontweight="bold")
        ax_pred.grid(axis="x", linestyle="--", alpha=0.2)
        for bar, val in zip(bars, values):
            ax_pred.text(
                val + 1.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontweight="bold",
                color="#f8fafc",
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
    st.markdown("### Ringkasan Performa Model pada Data Uji")

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
        cmap="Blues",
        ax=axes[0],
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
    )
    axes[0].set_title("Confusion Matrix — Logistic Regression")

    feat_imp = (
        pd.Series(rf_model.feature_importances_, index=model_columns)
        .sort_values(ascending=False)
        .head(10)
    )
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[1], palette="magma")
    axes[1].set_title("Top 10 Faktor Paling Berpengaruh terhadap Churn")

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: METODOLOGI =================
with tab_method:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### Alur Pemrosesan & Konversi Mata Uang")
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
