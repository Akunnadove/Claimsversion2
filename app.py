import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page config
st.set_page_config(page_title="Data Quality Classification", layout="wide")

# Title
st.title("📊 Data Quality Classification System")
st.markdown("Classify insurance claims into Priority, Abnormal, or Normal categories")

# Load the trained model
try:
    model = pickle.load(open('classification_model.pkl', 'rb'))
except:
    st.error("Model file not found. Please train the model first using classification_model.py")
    st.stop()

# ============================================================================
# STEP 1: UPLOAD DATA
# ============================================================================

st.sidebar.header("Upload Data")
upload_option = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Entry"])

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} records")
    else:
        st.info("Please upload a CSV file to get started")
        st.stop()

else:  # Manual Entry
    st.sidebar.subheader("Enter Record Details")
    
    data_dict = {
        'source_sheet': st.sidebar.text_input("Source Sheet", ""),
        'year': st.sidebar.number_input("Year", min_value=1990, max_value=2100, value=2024),
        'month': st.sidebar.number_input("Month", min_value=1, max_value=12, value=1),
        'guarantee_declaration_nr': st.sidebar.text_input("Guarantee Declaration Nr", ""),
        'POLICY_NR': st.sidebar.text_input("Policy Number", ""),
        'PERSON_NR': st.sidebar.text_input("Person Number", ""),
        'INSURED_NR': st.sidebar.text_input("Insured Number", ""),
        'GENDER': st.sidebar.selectbox("Gender", ["M", "F", "Other"]),
        'AGE': st.sidebar.number_input("Age", min_value=0, max_value=120, value=40),
        'general_practitioner': st.sidebar.text_input("General Practitioner", ""),
        'pzs_doctor_last_name': st.sidebar.text_input("PZS Doctor Last Name", ""),
        'pzs_doctor_first_name': st.sidebar.text_input("PZS Doctor First Name", ""),
        'beneficiary_provider_nr': st.sidebar.text_input("Beneficiary Provider Nr", ""),
        'provider_type': st.sidebar.text_input("Provider Type", ""),
        'description': st.sidebar.text_input("Description", ""),
        'beneficiary_provider_last_name': st.sidebar.text_input("Beneficiary Last Name", ""),
        'beneficiary_provider_first_name': st.sidebar.text_input("Beneficiary First Name", ""),
        'treatment_code': st.sidebar.text_input("Treatment Code", ""),
        'treatment_description': st.sidebar.text_input("Treatment Description", ""),
        'number_of_treatments': st.sidebar.number_input("Number of Treatments", min_value=0, value=1),
        'claim_date': st.sidebar.date_input("Claim Date", datetime.now()),
        'claim_amount': st.sidebar.number_input("Claim Amount", min_value=0.0, value=1000.0),
        'currency_description': st.sidebar.selectbox("Currency", ["EUR", "USD", "GBP"]),
        'payment_date': st.sidebar.date_input("Payment Date", datetime.now()),
        'entry_date': st.sidebar.date_input("Entry Date", datetime.now()),
        'audit_date': st.sidebar.date_input("Audit Date", datetime.now()),
        'tariff_code': st.sidebar.text_input("Tariff Code", ""),
        'description.1': st.sidebar.text_input("Description (2)", ""),
        'PAYMENT_ORDER_NR': st.sidebar.text_input("Payment Order Nr", ""),
    }
    
    df = pd.DataFrame([data_dict])

# ============================================================================
# STEP 2: CALCULATE FLAGS
# ============================================================================

# Convert date columns to datetime if they're strings
date_columns = ['claim_date', 'payment_date', 'entry_date', 'audit_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Flag 1: Age
df['flag_age'] = (df["AGE"] < 10) | (df["AGE"] > 75)

# Flag 2: Duplicates
df['flag_dup'] = df.duplicated(subset=[
    'source_sheet', 'year', 'month', 'guarantee_declaration_nr', 'POLICY_NR', 
    'PERSON_NR', 'INSURED_NR', 'GENDER', 'AGE', 'general_practitioner', 
    'pzs_doctor_last_name', 'pzs_doctor_first_name', 'beneficiary_provider_nr', 
    'provider_type', 'description', 'beneficiary_provider_last_name', 
    'beneficiary_provider_first_name', 'treatment_code', 'treatment_description', 
    'number_of_treatments', 'claim_date', 'claim_amount', 'currency_description', 
    'payment_date', 'entry_date', 'audit_date', 'tariff_code', 'description.1', 
    'PAYMENT_ORDER_NR'
], keep=False)

# Flag 3: Date logic
df['flag_date'] = (df["payment_date"] < df["claim_date"]).fillna(False)

# Flag 4: Outliers
Q1 = df["claim_amount"].quantile(0.25)
Q3 = df["claim_amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['flag_outlier'] = (df["claim_amount"] < lower_bound) | (df["claim_amount"] > upper_bound)

# Count flags
df['num_flags'] = df['flag_age'].astype(int) + df['flag_dup'].astype(int) + df['flag_date'].astype(int) + df['flag_outlier'].astype(int)

# Classify
df["quality_class"] = np.where(df['num_flags'] == 4, "Priority", 
                    np.where(df['num_flags'] > 0, "Abnormal", "Normal"))

# ============================================================================
# STEP 3: PREDICT WITH MODEL
# ============================================================================

# Prepare features for model (same columns used during training)
numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                and col not in ['quality_class', 'num_flags']]

X = df[numeric_cols].copy()
X = X.fillna(X.mean())

# Make predictions
df["predicted_class"] = model.predict(X)
df["prediction_confidence"] = model.predict_proba(X).max(axis=1)

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Details", "Flags", "Data"])

with tab1:
    st.subheader("Classification Results")
    
    cols = st.columns(len(df))
    
    for idx, (col, row) in enumerate(zip(cols, df.itertuples())):
        with col:
            # Color based on classification
            if row.quality_class == "Priority":
                color = "🔴"
            elif row.quality_class == "Abnormal":
                color = "🟠"
            else:
                color = "🟢"
            
            st.metric(
                label=f"Record {idx + 1}",
                value=f"{color} {row.quality_class}",
                delta=f"Predicted: {row.predicted_class}",
                delta_color="off"
            )

with tab2:
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        priority_count = (df["quality_class"] == "Priority").sum()
        st.metric("Priority", priority_count, f"{priority_count/len(df)*100:.1f}%")
    
    with col3:
        abnormal_count = (df["quality_class"] == "Abnormal").sum()
        st.metric("Abnormal", abnormal_count, f"{abnormal_count/len(df)*100:.1f}%")
    
    with col4:
        normal_count = (df["quality_class"] == "Normal").sum()
        st.metric("Normal", normal_count, f"{normal_count/len(df)*100:.1f}%")
    
    # Classification distribution chart
    st.subheader("Classification Distribution")
    class_counts = df["quality_class"].value_counts()
    st.bar_chart(class_counts)
    
    # Prediction confidence
    st.subheader("Prediction Confidence")
    st.write(f"Average Confidence: {df['prediction_confidence'].mean():.2%}")
    st.write(f"Min Confidence: {df['prediction_confidence'].min():.2%}")
    st.write(f"Max Confidence: {df['prediction_confidence'].max():.2%}")

with tab3:
    st.subheader("Flag Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age Flags", df['flag_age'].sum())
    
    with col2:
        st.metric("Duplicate Flags", df['flag_dup'].sum())
    
    with col3:
        st.metric("Date Logic Flags", df['flag_date'].sum())
    
    with col4:
        st.metric("Outlier Flags", df['flag_outlier'].sum())
    
    st.subheader("Flag Details by Record")
    flag_df = df[['POLICY_NR', 'flag_age', 'flag_dup', 'flag_date', 'flag_outlier', 'num_flags', 'quality_class', 'predicted_class']].copy()
    st.dataframe(flag_df, use_container_width=True)

with tab4:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )

# ============================================================================
# STEP 5: CONDITION EXPLANATIONS
# ============================================================================

st.markdown("---")
st.subheader("📋 Classification Conditions")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🔴 Age Flag", expanded=False):
        st.write("**Triggered when:** Age < 10 or Age > 75")
        st.write("Identifies unusual ages that may indicate data entry errors.")
    
    with st.expander("🔴 Duplicate Flag", expanded=False):
        st.write("**Triggered when:** Exact duplicate rows found")
        st.write("Identifies duplicate records that inflate metrics.")
    
with col2:
    with st.expander("🔴 Date Logic Flag", expanded=False):
        st.write("**Triggered when:** Payment Date < Claim Date")
        st.write("Identifies logically impossible date sequences.")
    
    with st.expander("🔴 Outlier Flag", expanded=False):
        st.write("**Triggered when:** Claim Amount outside IQR bounds")
        st.write(f"Current bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        st.write("Identifies statistical outliers using IQR method (1.5 × IQR).")

st.markdown("---")
st.markdown("""
**Classification Legend:**
- 🟢 **Normal**: No flags triggered (0 conditions)
- 🟠 **Abnormal**: 1-3 flags triggered (some conditions)
- 🔴 **Priority**: All 4 flags triggered (all conditions)
""")
