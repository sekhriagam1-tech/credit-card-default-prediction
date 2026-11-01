
import streamlit as st
import pandas as pd
import joblib

# Load model
saved = joblib.load("credit_default_model.pkl")
model = saved["model"]
features = saved["features"]

st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.stApp {
    background-color: #f1f5fa;
}
h1 {
    color: #12345a;
    text-align: center;
}
h2, h3 {
    color: #12345a;
}
.stButton button {
    background-color: #123f70;
    color: white;
    width: 100%;
    height: 50px;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Default Prediction")

st.markdown(
    "<p style='text-align:center;'>"
    "AI Based Credit Risk Assessment System"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

st.info("Please enter the customer's details below.")

with st.form("credit_form"):

    # PERSONAL INFORMATION
    st.header("👤 Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        limit_bal = st.number_input(
            "Credit Limit",
            min_value=0,
            value=50000
        )

    with c2:
        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with c3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    c1, c2 = st.columns(2)

    with c1:
        education = st.selectbox(
            "Education",
            ["Graduate School", "University",
             "High School", "Others"]
        )

    with c2:
        marriage = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Others"]
        )

    # REPAYMENT HISTORY
    st.divider()
    st.header("📅 Repayment History")

    pay_values = {}

    payment_options = {
        "Paid duly (-2)": -2,
        "No consumption (-1)": -1,
        "No delay (0)": 0,
        "1 month delay": 1,
        "2 months delay": 2,
        "3 months delay": 3,
        "4 months delay": 4,
        "5 months delay": 5,
        "6 months delay": 6,
        "7 months delay": 7,
        "8 months delay": 8
    }

    payment_months = {
        "PAY_0": "September",
        "PAY_2": "August",
        "PAY_3": "July",
        "PAY_4": "June",
        "PAY_5": "May",
        "PAY_6": "April"
    }

    cols = st.columns(3)

    for i, (key, month) in enumerate(payment_months.items()):
        with cols[i % 3]:
            selected = st.selectbox(
                month + " Payment Status",
                list(payment_options.keys()),
                key=key
            )
            pay_values[key] = payment_options[selected]

    # BILL AMOUNTS
    st.divider()
    st.header("💰 Previous Bill Amounts")

    bill_values = {}

    bill_months = {
        "BILL_AMT1": "September",
        "BILL_AMT2": "August",
        "BILL_AMT3": "July",
        "BILL_AMT4": "June",
        "BILL_AMT5": "May",
        "BILL_AMT6": "April"
    }

    cols = st.columns(3)

    for i, (key, month) in enumerate(bill_months.items()):
        with cols[i % 3]:
            bill_values[key] = st.number_input(
                month + " Bill Amount",
                min_value=0,
                value=10000,
                step=1000,
                key=key
            )

    # PREVIOUS PAYMENTS
    st.divider()
    st.header("💳 Previous Payment Amounts")

    amount_values = {}

    cols = st.columns(3)

    for i in range(1, 7):
        key = "PAY_AMT" + str(i)

        with cols[(i-1) % 3]:
            amount_values[key] = st.number_input(
                "Payment Amount " + str(i),
                min_value=0,
                value=2000,
                step=500,
                key=key
            )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Credit Default"
    )

# PROCESS INPUT AND DISPLAY RESULT
if submitted:

    sex_code = 1 if sex == "Male" else 2

    education_codes = {
        "Graduate School": 1,
        "University": 2,
        "High School": 3,
        "Others": 4
    }

    marriage_codes = {
        "Married": 1,
        "Single": 2,
        "Others": 3
    }

    user_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex_code,
        "EDUCATION": education_codes[education],
        "MARRIAGE": marriage_codes[marriage],
        "AGE": age
    }

    user_data.update(pay_values)
    user_data.update(bill_values)
    user_data.update(amount_values)

    # Match original training columns
    input_df = pd.DataFrame(
        [[user_data.get(f, 0) for f in features]],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    default_probability = (
        probabilities[classes.index(1)] * 100
    )

    st.divider()
    st.header("📊 Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        if prediction == 1:
            st.error("⚠️ Likely to Default")
        else:
            st.success("✅ Unlikely to Default")

    with c2:
        st.metric(
            "Estimated Default Probability",
            f"{default_probability:.2f}%"
        )

    st.progress(
        int(round(default_probability))
    )

    st.warning(
        "This prediction is an experimental model estimate. "
        "It is not a financial decision or a guarantee of future behavior."
    )
