# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:31:00 2019

@author: HA5035615
"""

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

##########Initial try on Regression and then move on to Logistic regression 

X, y = make_classification(n_samples=10, n_features=1, n_informative=1, n_redundant=0 , n_clusters_per_class=1, flip_y=0, random_state=7)


plt.figure(figsize=(10,6))
plt.scatter(X, y, c='red', marker='x')
plt.ylabel("Malignant Tumor {1: Yes  0: No}")
plt.xlabel("Tumor Size")
plt.show()


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)



z = np.linspace(-2, 2, 1000)
z_predict = lm.intercept_ + (lm.coef_ * z)

plt.figure(figsize=(10,6))
plt.scatter(X, y, c='red', marker='x')
plt.plot(z, z_predict)
plt.ylabel("Presence of Cancer {1: Yes  0: No}")
plt.xlabel("Tumor Size")
plt.show()


## now tracing horizontal and vertical line 

x_critical = (0.5 - lm.intercept_)/ lm.coef_     #( y = mx + c ) where c is intercept , m is coefficient and y = 0.5 whcih we want

plt.figure(figsize=(10,6))
plt.scatter(X, y, c='r', marker='x')
plt.plot(z, z_predict)
plt.axvline(x=x_critical, color='r', linestyle='--')
plt.axhline(y=0.5, color='g', linestyle='--')
plt.ylabel("Presence of Cancer {1: Yes  0: No}")
plt.xlabel("Tumor Size")



#### Now lets explain the issue of this if we have a outlier 


new_obs = 20     ## adding outlier 
X = np.vstack([X, new_obs])    # stack up outlier 
y = np.append(y, 1)     # stach up outlier outcome 
lm = LinearRegression()
lm.fit(X, y)
z = np.linspace(-2, new_obs, 1000)
z_predict = lm.intercept_ + (lm.coef_ * z)
x_critical2 = (0.5 - lm.intercept_)/ lm.coef_

              
              
plt.figure(figsize=(10,6))
plt.scatter(X, y, c='r', marker='x')
plt.plot(z, z_predict)
plt.axvline(x=x_critical, color='r', linestyle='--')
plt.axvline(x=x_critical2, color='y', linestyle='--')
plt.ylabel("Presence of Cancer {1: Yes  0: No}")
plt.xlabel("Tumor Size")
plt.show()          
    

#############Now on Logistic Regression 

from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=10, n_features=1, n_informative=1, n_redundant=0 , n_clusters_per_class=1, flip_y=0, random_state=7)
clf = LogisticRegression()
clf.fit(X, y)



plt.figure(figsize=(10,6))
plt.scatter(X, y, c='r', marker='x')
xt = np.linspace(-3, 3, 1000).reshape(1000,1)
yt = clf.predict(xt)
plt.plot(xt, yt)
plt.ylabel("Presence of Cancer {1: Yes  0: No}")
plt.xlabel("Tumor Size")



## for an outlier problem 


# append an outlier 
new_obs = 20
X = np.vstack([X, new_obs])
y = np.append(y, 1)
clf.fit(X, y)

plt.figure(figsize=(10,6))
plt.scatter(X, y, c='r', marker='x')
xt = np.linspace(-5, 25, 1000).reshape(1000,1)
yt = clf.predict(xt)
plt.plot(xt, yt)
plt.xlabel('Feature')
plt.ylabel('Target')



############# Another problem from load prediction data 


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

dataframe = pd.read_csv(r'D:\Others\PythonCode\MLinAction\Log reg data\loanprediction.csv')
X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
logistic_regressor = LogisticRegression()
pipeline = Pipeline(steps=[('add_poly_features', PolynomialFeatures()),('logistic_regression', logistic_regressor)])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print (accuracy_score(y_test, y_pred))