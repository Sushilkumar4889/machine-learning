#Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data Set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the Dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_Transform(X_train)
X_test = sc_Y.transform(X_test)
sc_Y =StandardScaler()
Y_train = sc.Y.fit_transform(Y_train)'''

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test Set Results
Y_pred = regressor.predict(X_test)

#Visualising the Training set Results
'''plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()'''

#Visualising the Test Set Results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
