import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models import predict_cardio

# get values from functions
features, target, scaler, model, Y_pred, cr, cm = predict_cardio()

st.title('Cardiovascular Disease Prediction', text_alignment='left')
st.subheader('_Using_ :red[Logistic] _Regression_ :blue[Model]', text_alignment='left')

st.sidebar.header(
    '_Features_'
)


age = st.sidebar.slider(
    'Age',
    min_value=27,
    max_value=70,
    value=35,
    step=1
)

# radio button
gender_options = {1: 'Female', 2: 'Male'}
gender = st.sidebar.radio(
    'Gender',
    options = list(gender_options.keys()),
    format_func = lambda values : gender_options.get(values)
)

height = st.sidebar.slider(
    'Height',
    min_value=120,
    max_value=207,
    value=130,
    step=1
)

weight = st.sidebar.slider(
    'Weight',
    min_value=40,
    max_value=140,
    value=70,
    step=1    
)

ap_hi = st.sidebar.slider(
    'Systolic Pressure',
    min_value=100,
    max_value=200,
    value=120,
    step=1
)

ap_lo = st.sidebar.slider(
    'Di-Systolic Pressure',
    min_value=50,
    max_value=90,
    value=80,
    step=1
)

cholesterol_options = {1: 'Healthy', 2: 'Mild', 3:'High Cholesterol'}
cholesterol = st.sidebar.selectbox(
    'Cholesterol',
    options = list(cholesterol_options.keys()),
    format_func = lambda values : cholesterol_options.get(values)
) 

gluc_options = {1: 'Healthy', 2: 'Mild', 3:'High Glucose'}
gluc = st.sidebar.selectbox(
    'Glucose',
    options = list(gluc_options.keys()),
    format_func = lambda values : gluc_options.get(values)
) 

smoke_options = {0: 'Doesnot Smoke', 1: 'Does Smoke'}
smoke = st.sidebar.radio(
    'Smoke',
    options = list(smoke_options.keys()),
    format_func = lambda x : smoke_options.get(x)
)

# 'alco', 
alco_options = {0: 'Doesnot Drink Alcohol', 1: 'Does Drink Alcohol'}
alco = st.sidebar.radio(
    'Alcohol Consumption',
    options = list(alco_options.keys()),
    format_func = lambda x : alco_options.get(x)
)

# 'active'
active_options = {0: 'Doesnot do PA', 1: 'Does PA'}
active = st.sidebar.radio(
    'Physical Activities',
    options = list(active_options.keys()),
    format_func = lambda x : active_options.get(x)
)

# button
if st.button('Predict Cardio'):
    input_data = pd.DataFrame([[
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    ]], columns=features)
    
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)
    
    if prediction[0] == 0:
        st.write('No cardiovascular disease found!!')
        st.success('Patient is likely to be healthy.')
    else:
        st.write('Cardiovascular disease found!!')
        st.warning('Patient is likely to be unhealthy.')

# Visualization
st.header('Visualization')
st.subheader('Confusion Matrix')

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=['Predicted Healthy [0]', 'Predicted UnHealthy[1]'],
           yticklabels=['Actual Healthy [0]', 'Actual UnHealthy[1]'])
st.pyplot(fig)

st.subheader('Classification Report')
# change row to column and column to row
cr_df = pd.DataFrame(cr).transpose()
st.dataframe(cr_df.style.format(precision=2))