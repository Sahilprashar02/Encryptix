# SAHIL PRASHAR


        # TASK 4 : Sales Prediction 

# Importing Neccessary Libraries
import pandas as pd        # csv data manipulation
import streamlit as st     # user interface
import seaborn as sb       # for visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('Advertising.csv')

st.title(":violet[Sales Prediction using Python]")
st.write("---")

st.info("Sales Prediction involves forecasting the amount of a product that customers will purchase,taking into various factors such as Tv, Radio, Newspaper Advertising Cost.")

st.write("Used Linear Regression Algorithm to analyze and interpret data, allowing user to make informed decisions regarding advertising costs.")

# making tabs for different things
t1,t2,t3,t4,t5,t6 = st.tabs(["Dataset","Insights","Pairplot","Histogram","Heatmap","Prediction"])

with t1:
        st.info(("Given Dataset is consist of Advertisting platform and the related Sales."))
        st.subheader("Head of the Dataset")
        st.write(data.head())    # display the dataset

with t2:
        st.subheader(":grey[Getting insights into Dataset in Detail]")
        st.write(data.describe())

        st.subheader(":grey[Analysis of this Dataset]")
        st.info(":green[Average Expenses spend is Highest on TV]")
        st.info(":red[Average Expenses spend is Lowes on Radio]")
        st.info(":green[Maximum sales is 27 and Minimum sales is 1.6]")

with t3:
        # Data visualization with pairplot graph
        st.subheader("Pairplot Graphs")
        sb.pairplot(data, x_vars=["TV","Radio","Newspaper"], y_vars="Sales", kind="scatter")
        st.pyplot(plt)
        plt.close()
        st.subheader(":green[Pairplot Analysis]" )
        st.info("When Advertising cost increases for TV Ads, the Sales is also increases but for Radio and Newspaper it is quit unpredictable")

with t4:
       # Visualize histogram for each platform
       tabs = st.tabs(["TV","Radio","Newspaper"])

       with tabs[0]:
                st.subheader("Histograms Chart For TV")
                plt.figure(figsize=(10,4))
                data['TV'].plot.hist(bins=10,color="grey")
                plt.xlabel("TV")
                plt.ylabel("Sales")
                plt.title("Sales based on TV Advertising cost")
                st.pyplot(plt)
                plt.close()
       with tabs[1]:
                st.subheader("Histogram Chart For Radio")
                plt.figure(figsize=(10,4))
                data['Radio'].plot.hist(bins=10,color="#a6bddb")
                plt.xlabel("Radio")
                plt.ylabel("Sales")
                plt.title("Sales based on Radio Advertising cost")
                st.pyplot(plt)
                plt.close()
       with tabs[2]:
                st.subheader("Histogram Chart For Newspaper")
                plt.figure(figsize=(10,4))
                data["Newspaper"].plot.hist(bins=10,color="#99d8c9")
                plt.xlabel("Newspaper")
                plt.ylabel("Sales")
                plt.title("Sales based on Newspaper Advertising Cost")
                st.pyplot(plt)
                plt.close()

       st.subheader(":green[Histogram Analysis]" )
       st.info("Lowest Advertising code is for Newspaper")

with t5:
        # Data visualization using heatmap
        sb.heatmap(data.corr(),annot=True)
        plt.title("Heatmap Analysis For All kind of Newspaper Cost")
        st.pyplot(plt)
        plt.close()

        st.subheader(":green[Heatmap Analysis]")
        st.info("Highets Coorelated Sales is For TV")

with t6:
           # load model, fit it into data, and predict 
        st.write("---")
        st.subheader("Train the Model using Linear Regression.")
        st.write("Taking only one Advertisig variable which is TV because It is Highly Coorelated with sales")
        st.write("---")

        import numpy as np
        x = data['TV']
        y = data['Sales']       

        # splitting data into training and testing part
        x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2)

        x_train = np.array(x_train).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)

        from sklearn.linear_model import LinearRegression

        # loading the model
        model = LinearRegression()

        # fit the training data into model
        model.fit(x_train,y_train)

        plt.plot()
        # taking user input for
        a = st.number_input("Enter any TV Advertising Cost ")

        if st.button("Make Sales Prediction"):
        
            a = np.array(a).reshape(-1,1)
            y_pred = model.predict(a)
            st.subheader(y_pred)



