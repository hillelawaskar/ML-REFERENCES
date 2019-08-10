# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:46:13 2019




# if you see this data you will find that the actuals start deviating from predictions on higher values ; so we need to 
# consider a polynominal ; so you need to do feature scale , i.e. one of the feature you need to square ;
# for polonominial ; with quare ; you wull get a downside U as graph 
# If you add cube then it will be halh downside U and then it will start going up ; so we need to use feature scaling 

      

"""

##############################################
# Polynominal regression 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Importing the dataset 
datas = pd.read_csv(r'D:\Others\PythonCode\MLinAction\Poly data\data.csv') 
datas 


X = datas.iloc[:, 1:2].values 
y = datas.iloc[:, 2].values
              
              
# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 
  
lin.fit(X, y) 


# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show()


# Predicting a new result with Linear Regression 
lin.predict(110.0) 


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4)  # this is teh number of polinomial variables to add 
X_poly = poly.fit_transform(X)  # create that order matrix
print (X_poly)  

poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y)

# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show()

# Predicting a new result with Polynomial Regression 
lin2.predict(poly.fit_transform(110.0)) 



