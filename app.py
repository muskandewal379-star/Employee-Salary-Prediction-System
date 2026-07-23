import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary Prediction", page_icon="💼")

st.title(" 💼 Employee Salary Prediction App")

df = pd.read_csv("salary.csv")

print(df.columns)
print(df.head())

#input
x = df[["Experience"]]

#output
y = df["Salary"]

print(df.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression ()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

r2 = r2_score(y_test, prediction)

st.success(f" Our Model Accuracy is: {(r2*100):.2f}%")
Experience = 500

Experience = st.sidebar.number_input("Enter your Experience in years:", min_value=5, max_value=10, value=5, step=1)
Salary = st.sidebar.number_input("Enter your Salary:", min_value=10000, max_value=100000, value=50000, step=5000)

if st.sidebar.button("Check Now"):
    result = model.predict([[Experience,Salary]])
    approved = 0
    if result[0] == 1:
        approved = min(Salary, int(Experience * 8 + (Salary - 650) * 400 + Experience * 1000))
        if approved < 0:
            approved = 0
        st.success("Salary Approved")
        st.sidebar.warning(f"Approved Salary Amount:₹ {approved}")
    else:
        st.error("Salary Rejected")