# SAHIL PRASHAR
                # TASK 1 : Titanic Servival Prediction


import pandas as pd  # for loading csv
from sklearn.model_selection import train_test_split # spliting traing and testing data
from sklearn.naive_bayes import GaussianNB # model that predict

import streamlit as st  # gives graphical interface

df = pd.read_csv("Titanic.csv")  # reading csv file
print(df.head())

# Removing Unneccsary columns that are not useful for prediction
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

# making binary data for gender bcz it is not balanced so 0 for male and 1 for female
d = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(d)  # Note: 'Sex' is the column name in Titanic dataset
print(df.head())

target = df.Survived  # dependent variable
print(target)

input_var = df.drop('Survived', axis='columns')  # independent variables

# Handle missing values for the 'Age' column
input_var.Age = input_var.Age.fillna(input_var.Age.mean())
print(input_var.head(10))

st.header(":blue[Titanic Survival Prediction]")

# Split the dataset into training and testing parts
x_train, x_test, y_train, y_test = train_test_split(input_var, target, test_size=0.2)

# Initialize the model
model = GaussianNB()

# Load training data into the model
model.fit(x_train, y_train)

# Testing the model
print(model.score(x_test, y_test))
print(y_test[:20])
print(model.predict(x_test)[:20])  # predicting x_test data based on y_test

# Getting input from the user
in_class = st.number_input("Enter Passenger Class: ")
gender = st.selectbox("Enter Gender: ", options=['male', 'female'])

if gender == 'male':
    in_gender = 0
else:
    in_gender = 1

in_age = st.number_input('Enter Age: ')
in_fare = st.number_input('Enter Fare: ')
in_age = float(in_age)
in_fare = float(in_fare)

test = pd.DataFrame({'Pclass': [in_class], 'Sex': [in_gender], 'Age': [in_age], 'Fare': [in_fare]})

a = model.predict(test)  # predicting testing data

if st.button("Make Prediction"):
    # if a==0 that means not survived else survived
    if a == 0:
        st.subheader("Passenger will Not Survive")
    else:
        st.subheader('Passenger will Survive')
