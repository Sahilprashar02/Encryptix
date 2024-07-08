# SAHIL PRASHAR

                    # TASK 3 : IRIS Flower Classification


# Importing neccessary Libraries
import pandas as pd
from sklearn.model_selection import (train_test_split)
import streamlit as st

data = pd.read_csv('IRIS.csv')  # Loading the dataset
print(data.head())

st.title("IRIS Flower Classification")

data = data.values

# slicing First 4 columns as input : sepal_length, sepal_width, petal_length, petal_width
inputs = data[:,0:4]    

# last column for targeting : class
op = data[:,4]

# spliting dataset into training and testing part
x_train,x_test,y_train,y_test = train_test_split(inputs,op,test_size=0.2)

# importing svc
from sklearn.svm import SVC

# load the model
model = SVC()

# train the model with training data
model.fit(x_train,y_train)

st.info("Enter feature such as sepal and petal length and width for predicting Flower Class.")

# Taking inputs
sepal_len = st.number_input('Enter Sepal Length : ')
sepal_wid = st.number_input('Enter Sepal Width : ')
petal_len = st.number_input('Enter Petal Length : ')
petal_wid = st.number_input('Enter Petal Width')
# merge the inputs in dataframe for prediction
a = pd.DataFrame({'sepal_length':[sepal_len],'sepal_width':[sepal_wid],'petal_length':[petal_len],'petal_width':[petal_wid]})
# Predicting
ans1 = model.predict(a)
btn = st.button('Make Flower Prediction')
if btn:
    st.subheader(ans1)


# accuray of model
acc_btn = st.button("Find Accuracy of model ")
    
if acc_btn:
    
    predict1 = model.predict(x_test)
    from sklearn.metrics import accuracy_score
    st.subheader("Accuracy Score")
    st.subheader(accuracy_score(y_test,predict1)*100)
    
    # Classification report of model
class_report = st.button("Find Classification Report")
    
if class_report:
    
    predict2 = model.predict(x_test)
    from sklearn.metrics import classification_report
    st.subheader("Classification Report of Model")
    st.write(classification_report(y_test,predict2)) 

