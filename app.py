app_code = """import streamlit as st
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

st.set_page_config(page_title="Prediksi Telco Customer Churn", layout="wide")
st.title("📊 Prediksi Pelanggan Berhenti (Telco Churn)")
st.caption("Aplikasi prediksi churn pelanggan telekomunikasi berbasis Machine Learning (Mata Uang: Rupiah)")

@st.cache_resource
def load_assets():
    url = 'https://raw.githubusercontent.com/fathirgans95/telco-customer-churn/main/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    y = df['Churn'].map({'No': 0, 'Yes': 1})
    X_encoded = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model, scaler, list(X_train.columns), X_test, y_test

log_model, rf_model, scaler, model_columns, X_test, y_test = load_assets()

tab1, tab2 = st.tabs(["🔮 Prediksi Pelanggan", "📈 Evaluasi & Grafik Model"])

with tab1:
    st.subheader("Input Data Pelanggan")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure / Lama Berlangganan (Bulan)", 1, 72, 12)

    with col2:
        contract = st.selectbox("Jenis Kontrak (Contract)", ["Month-to-month", "One year", "Two year"])
        phone_service = st.selectbox("Layanan Telepon (Phone Service)", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Layanan Internet (Internet Service)", ["Fiber optic", "DSL", "No"])

    with col3:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges_idr = st.number_input(
            "Tagihan Bulanan (Rp)", min_value=0, max_value=5000000, value=1120000, step=50000
        )
        total_charges_idr = st.number_input(
            "Total Tagihan (Rp)", min_value=0, max_value=200000000, value=int(tenure * monthly_charges_idr), step=100000
        )

    st.caption(f"Ringkasan Tagihan: **Rp {monthly_charges_idr:,.0f} / bulan** | Total: **Rp {total_charges_idr:,.0f}**".replace(",", "."))

    if st.button("Prediksi Churn", use_container_width=True):
        # Konversi Rupiah ke skala USD dataset agar prediksi model tetap akurat
        monthly_usd = monthly_charges_idr / KURS_IDR
        total_usd = total_charges_idr / KURS_IDR

        input_dict = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': tech_support,
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
            'PaperlessBilling': 'Yes', 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_usd, 'TotalCharges': total_usd
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

        pred = log_model.predict(input_encoded)[0]
        proba = log_model.predict_proba(input_encoded)[0]
        prob_stay, prob_churn = proba[0] * 100, proba[1] * 100

        st.divider()
        st.subheader("Hasil Prediksi")

        res_col1, res_col2 = st.columns([1, 1.3])
        with res_col1:
            if pred == 1:
                st.error(f"⚠️ Berisiko Churn (Yes) — Probabilitas Churn: {prob_churn:.1f}%")
            else:
                st.success(f"✅ Cenderung Bertahan (No) — Probabilitas Churn: {prob_churn:.1f}%")
            st.metric("Peluang Bertahan (No)", f"{prob_stay:.1f}%")
            st.metric("Peluang Churn (Yes)", f"{prob_churn:.1f}%")

        with res_col2:
            fig_pred, ax_pred = plt.subplots(figsize=(5.5, 2.8))
            labels = ['Bertahan (No)', 'Berisiko Churn (Yes)']
            values = [prob_stay, prob_churn]
            colors = ['#2ecc71', '#e74c3c']
            bars = ax_pred.barh(labels, values, color=colors, height=0.5)
            ax_pred.set_xlim(0, 108)
            ax_pred.set_xlabel('Probabilitas (%)')
            ax_pred.set_title('Visualisasi Probabilitas Pelanggan Ini')
            for bar, val in zip(bars, values):
                ax_pred.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_pred)

with tab2:
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LogReg Accuracy", f"{accuracy_score(y_test, y_pred_log)*100:.2f}%")
    m2.metric("LogReg ROC-AUC", f"{roc_auc_score(y_test, y_prob_log):.4f}")
    m3.metric("RF Accuracy", f"{accuracy_score(y_test, y_pred_rf)*100:.2f}%")
    m4.metric("RF ROC-AUC", f"{roc_auc_score(y_test, y_prob_rf):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cm = confusion_matrix(y_test, y_pred_log)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
    axes[0].set_title('Confusion Matrix - Logistic Regression')

    feat_imp = pd.Series(rf_model.feature_importances_, index=model_columns).sort_values(ascending=False).head(10)
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[1], palette='viridis')
    axes[1].set_title('Top 10 Faktor Penyebab Churn')

    plt.tight_layout()
    st.pyplot(fig)
"""

with open("app.py", "w") as f:
    f.write(app_code)

import subprocess, time, re
subprocess.Popen(["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"])
time.sleep(3)

tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

for line in tunnel.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        print("\n✅ LINK WEBSITE VERSI RUPIAH (Rp):")
        print(match.group(0))
        break
