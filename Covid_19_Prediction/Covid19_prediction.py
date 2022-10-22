# Databricks notebook source
!pip install openpyxl

# COMMAND ----------

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# COMMAND ----------

df = pd.read_excel('./Covid19_2020_India.xlsx',parse_dates=['Date'])

# COMMAND ----------

df.head()

# COMMAND ----------

# Keeping only required columns

df = df[['Date','Name of State / UT','Latitude','Longitude','Total Confirmed cases']]
df.head()

# COMMAND ----------

max_confirmed_cases_in_a_day = df.sort_values(by='Total Confirmed cases',ascending=False)
max_confirmed_cases_in_a_day.head()

# COMMAND ----------

state_with_maximum_number_of_cases = df.groupby(["Name of State / UT"]).sum()
state_with_maximum_number_of_cases = state_with_maximum_number_of_cases.sort_values(by='Total Confirmed cases',ascending=False).reset_index()
state_with_maximum_number_of_cases

# COMMAND ----------

# Making bar plot or states with top confirmed cases
sns.set(rc={'figure.figsize':(10,7)})
sns.barplot(x="Name of State / UT",y="Total Confirmed cases",data=state_with_maximum_number_of_cases)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# Covid Trend State Wise
for i in df["Name of State / UT"].unique():
    
    sns.set(rc={'figure.figsize':[15,10]})
    sns.lineplot(x="Date",y="Total Confirmed cases",data=df.loc[df["Name of State / UT"]==i].sort_values(["Date"]))
    plt.title(f"{i} covid trend")
    plt.show()

# COMMAND ----------

# Converting date-time to ordinal
df['Date']=df['Date'].map(dt.datetime.toordinal)
df.head()

# COMMAND ----------

df["Name of State / UT"] = df["Name of State / UT"].astype(str)
df["Total Confirmed cases"] = df["Total Confirmed cases"].astype(int)
df["Latitude"] = df["Latitude"].astype(float)
df["Longitude"] = df["Longitude"].astype(float)

# COMMAND ----------

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'states'.
df['Name of State / UT']= label_encoder.fit_transform(df['Name of State / UT'])
  
df['Name of State / UT'].unique()

# COMMAND ----------

df = df[:][:500]

# COMMAND ----------

# Getting dependent variable and independent variable
x = df.iloc[:,:-1]
y = df['Total Confirmed cases']

# COMMAND ----------

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# COMMAND ----------

lr = LinearRegression(normalize=True)
lr.fit(np.array(x_train),np.array(y_train))

# COMMAND ----------

print(lr.score(x_test, y_test))

# COMMAND ----------

y_pred = lr.predict(x_test)
plt.scatter(x_test["Date"], y_test, color ='b')
plt.plot(x_test["Date"], y_pred, color ='k')
  
plt.show()

# COMMAND ----------

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
  
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

# COMMAND ----------

print(lr.intercept_)

# COMMAND ----------

print(lr.coef_)

# COMMAND ----------


