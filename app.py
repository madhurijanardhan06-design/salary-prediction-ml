import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Experience': [1,2,3,4,5,6,7,8],
    'Salary': [25000,30000,35000,40000,50000,60000,70000,80000]
}

df = pd.DataFrame(data)

# Input and Output
X = df[['Experience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# App title
st.title("Salary Prediction Using Machine Learning")

# User input
exp = st.number_input("Enter Years of Experience")

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[exp]])
    st.success(f"Predicted Salary = ₹ {prediction[0]:,.0f}")