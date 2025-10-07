#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


# In[2]:


# Load and train model
iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier()
model.fit(X, y)



# In[4]:


# Save model
joblib.dump(model, r"C:\Users\vangala.ranadheer\Desktop\Ranadheer\Projects\Deployment\iris\iris_model.pkl")

