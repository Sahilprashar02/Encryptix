# Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Separate fraudulent and non-fraudulent transactions
non_fraud = data[data.Class == 0]
fraud = data[data.Class == 1]

# Save the separated datasets (optional)
non_fraud.to_csv('non_fraud.csv', index=False)
fraud.to_csv('fraud.csv', index=False)

# Balance the dataset by undersampling the non-fraudulent transactions
non_fraud_sample = non_fraud.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([non_fraud_sample, fraud], axis=0)

# Separate features and target variable
X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Check the performance of the model
train_accuracy = accuracy_score(model.predict(X_train), y_train)
test_accuracy = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("Credit Card Fraud Detection")
st.write("Enter Features to check if the transaction is legitimate or fraudulent")
st.info("Legitimate Transaction -> Non-Fraudulent Transaction (Normal Transaction)")

input_data = st.text_input("Enter Features (comma-separated)")

submit = st.button("Detect")

if submit:
    try:
        # Convert the input data into a numpy array
        input_values = np.array([float(i) for i in input_data.split(',')])

        if input_values.shape[0] != X.shape[1]:
            st.error("Invalid number of features entered. Please check the input.")
        else:
            # Reshape the input data and make a prediction
            detection = model.predict(input_values.reshape(1, -1))

            if detection[0] == 0:
                st.subheader("Legitimate Transaction")
            else:
                st.subheader("Fraudulent Transaction")
    except ValueError:
        st.error("Invalid input. Please enter numerical values only.")
