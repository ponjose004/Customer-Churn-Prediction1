import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts once
model = joblib.load('Model files/churn_model.pkl')
scaler = joblib.load('Model files/scaler.pkl')
encoders = joblib.load('Model files/label_encoders.pkl')

st.title('Customer Churn Prediction App')

# Containers for dynamic content
result_container = st.empty()
table_container = st.empty()

# Mode selector
mode = st.radio('Select mode:', ['Single Prediction', 'Batch Prediction'])

if mode == 'Single Prediction':
    # Clear batch output
    table_container.empty()

    # Single-input form
    with st.form('single_form'):
        st.subheader('Enter single customer details')
        credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
        geography = st.selectbox('Geography', encoders['Geography'].classes_)
        gender = st.selectbox('Gender', encoders['Gender'].classes_)
        age = st.number_input('Age', min_value=0, max_value=120, value=40)
        tenure = st.number_input('Tenure', min_value=0, value=3)
        balance = st.number_input('Balance', min_value=0.0, value=50000.0, format='%.2f')
        num_products = st.number_input('Number of Products', min_value=0, value=1)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active = st.selectbox('Is Active Member', [0, 1])
        salary = st.number_input('Estimated Salary', min_value=0.0, value=60000.0, format='%.2f')
        submit_single = st.form_submit_button('Predict')

    if submit_single:
        # Build DataFrame
        df = pd.DataFrame([{ 
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary
        }])
        # Encode and scale
        df['Geography'] = encoders['Geography'].transform(df['Geography'])
        df['Gender']    = encoders['Gender'].transform(df['Gender'])
        X = scaler.transform(df.values)
        # Predict
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0, 1]
        # Display immediately below the form
        pred_text = 'Will churn' if pred == 1 else 'Will not churn'
        st.markdown('---')
        st.subheader(pred_text)
        st.write(f"Probability of churn: {prob:.2%}")

elif mode == 'Batch Prediction':
    # Clear single output
    result_container.empty()

    uploaded_file = st.file_uploader('Upload CSV for batch prediction', type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Drop unwanted columns and target
        drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
        df_proc = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        # Impute numerics
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].median())
        # Encode
        df_proc['Geography'] = encoders['Geography'].transform(df_proc['Geography'])
        df_proc['Gender']    = encoders['Gender'].transform(df_proc['Gender'])
        # Scale
        X = scaler.transform(df_proc.values)
        # Predict
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        # Prepare output table
        output = pd.DataFrame({
            'CustomerId': df['CustomerId'],
            'Surname':    df['Surname'],
            'ChurnPrediction': ['Will churn' if p == 1 else 'Will not churn' for p in preds],
            'ChurnProbability': probs
        })
        table_container.dataframe(output)
