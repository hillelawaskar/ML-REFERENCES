# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:28:05 2019

@author: HA5035615
"""

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10, n_features=1, n_informative=1, n_redundant=0 , n_clusters_per_class=1, flip_y=0, random_state=7)




x1 = np.arange(6)
x2 = np.zeros_like(x1)
  
a = 0          




for xone in x1:
    print (xone)
    x2[xone] = 3- x1[xone]
    a += 1
    if a == 6:
        break
                

plt.plot(x1,x2)




y = 3




def p(x):
    return 4*x**2 + 3*x + 500


for x in [-1, 0, 2, 3.4]:
    print(x, p(x))
    
    
import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-10, 10, 50, endpoint=True)
F = p(X)
plt.plot(X,F)
plt.show()




#####################POLYNOMINALS

##Understand __repr__ : it is a printable representation like shown in the example below

class Point:
  def __init__(self, x, y):
    self.x, self.y = x, y
  def __repr__(self):
    return 'Point(x=%s, y=%s)' % (self.x, self.y)
p = Point(1, 2)
p
#Point(x=1, y=2)


##now lets get on to polynominals 
class Polynomial:
    
    def __init__(self, *coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0 
        """
        # for reasons of efficiency we save the coefficients in reverse order,
        # i.e. a_0, a_1, ... a_n
        self.coefficients = coefficients[::-1] # tuple is also turned into list
     
    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        # The internal representation is in reverse order,
        # so we have to reverse the list
        return "Polynomial" + str(self.coefficients[::-1])

    def __call__(self, x):    
        res = 0
        for index, coeff in enumerate(self.coefficients):
            res += coeff * x** index
        return res

p = Polynomial(4, 0, -4, 3, 0)
print(p)


for x in range(-10,10):
    print (x,p(x))



import matplotlib.pyplot as plt
X = np.linspace(-10, 10, 50, endpoint=True)
F = p(X)
plt.plot(X,F)
plt.show()
