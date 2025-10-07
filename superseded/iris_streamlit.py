#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(r"C:\Users\vangala.ranadheer\Desktop\Ranadheer\Projects\Deployment\iris\iris_model.pkl")

target_names = ['setosa', 'versicolor', 'virginica']

st.title("🌸 Iris Species Predictor")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)[0]
    species = target_names[prediction]
    st.success(f"🌼 Predicted Species: **{species.capitalize()}**")

