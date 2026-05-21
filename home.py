import streamlit as st

models = {
    "KNN": [
        st.Page('KNN.py', title='KNN Prediction'),
    ],
    "Logistic Regression": [
        st.Page('app.py', title='Logistic Prediction')
    ]
}

knn = st.navigation(models, position="top")
knn.run()