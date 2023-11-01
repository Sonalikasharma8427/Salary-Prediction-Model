import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#---------------IMPORTING LIBRARIES------------------

#-----------IMPORTING DATASET--------------
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#------------Spliting Dataset Into Training and Testting-----------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
# fit() method is going to train the model
lr.fit(x_train,y_train)

#-----------------Predict method-------------
y_pre =lr.predict(x_test)

#------------------Visualization of train set result-------------------
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')  #Enter lr.predict(x_train) get predicted values
plt.title('Salary Vs Experience(Training set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()


#------------------Visualization of test set result-------------------
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('Salary Vs Experience(Test set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

# print(lr.predict([[12]])) #Salary expect with 12 year of experience#