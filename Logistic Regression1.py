#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Logistic Regression
get_ipython().system('pip install sqlalchemy mysql-connector-python numpy pandas scikit-learn matplotlib')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_engine():
    engine = create_engine(
    "mysql+mysqlconnector://root:Ramu#143@127.0.0.1/linear_regression1"
    )
    return engine

def load_data():
    engine = get_engine()
    query = "select study_hrs, mock_score, result from student_performance"
    df = pd.read_sql(query, engine)
    return df

df = load_data()
print(df)

x = df[['study_hrs', 'mock_score']]
y = df[['result']]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Coefficients (w1, w2):", model.coef_)
print("Intercept (b):", model.intercept_)

new_student = [[3.5, 50]]
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("Prediction:", prediction)
print("Probability:", probability)




