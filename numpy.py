# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:36:57 2019

@author: HA5035615
"""

##########Numpy 

import numpy as np

a = np.arange(15).reshape(3, 5)
#Out[213]
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14]])
a.shape
#Out[211]: (3, 5)
a.ndim#
#Out[212]: 2

a.size
#Out[214]: 15

a.dtype
# dtype('int32')


a.itemsize
#Out[216] 4     #4 bytes for each element 

a.data    
#Out[218] <memory at 0x00000203E093E630>     # teh buffer containing actual data ; we dont use this as we will access wing indexing 


#Array creation    
a = np.array([2,3,4])
a.dtype
b = np.array([1.2, 3.5, 5.1])
b.dtype

#WRONG 
a = np.array(1,2,3,4)    # WRONG
#RIGHT 
a = np.array([1,2,3,4])  # RIGHT

            
b = np.array([(1.5,2,3), (4,5,6)])

c = np.array( [ [1,2], [3,4] ], dtype=complex )
c
#Out[232] 
#array([[1.+0.j, 2.+0.j],
#       [3.+0.j, 4.+0.j]])

np.zeros((3,4))
#array([[0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.]])
np.zeros((2,3,4))
#array([[[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]],
#
#       [[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]]])
np.empty( (2,3) )         # this is uninitialized
#array([[1.6502e-321, 2.1640e-321, 2.3715e-321],
#       [1.4318e-320, 1.4550e-320, 1.4713e-320]])
    
np.ones( (2,3,4), dtype=np.int16 )    
#array([[[1, 1, 1, 1],
#        [1, 1, 1, 1],
#        [1, 1, 1, 1]],
#
#       [[1, 1, 1, 1],
#        [1, 1, 1, 1],
#        [1, 1, 1, 1]]], dtype=int16)    

np.arange( 10, 30, 5 )     # start from 10 , less than 30 and jump by 5
#array([10, 15, 20, 25])

np.arange( 0, 2, 0.3 )                 # it accepts float arguments
#array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])

from numpy import pi
np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
#array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
#array([0.        , 0.06346652, 0.12693304, ..., 6.15625227, 6.21971879,
#       6.28318531]
f = np.sin(x)
#array([ 0.00000000e+00,  6.34239197e-02,  1.26592454e-01, ...,
#       -1.26592454e-01, -6.34239197e-02, -2.44929360e-16])


np.zeros_like(f).shape    # if you want to create ndarray like structure of some other wich are zeros 
#(100,)


a = np.arange(6)
#[0 1 2 3 4 5]

a.reshape(6,1)
#array([[0],
#       [1],
#       [2],
#       [3],
#       [4],
#       [5]])a

b = np.arange(12).reshape(4,3)

#[[ 0  1  2]
# [ 3  4  5]
# [ 6  7  8]
# [ 9 10 11]]


c = np.arange(24).reshape(2,3,4)
#array([[[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]],
#
#       [[12, 13, 14, 15],
#        [16, 17, 18, 19],
#        [20, 21, 22, 23]]])



a = np.array( [20,30,40,50] )
b = np.arange( 4 )
c = a-b
#array([20, 29, 38, 47])

b**2
#array([0, 1, 4, 9])

10*np.sin(a)
#array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])

a<35
# array([ True,  True, False, False])




A = np.array( [[1,1],
             [0,1]] )
B = np.array( [[2,0],
             [3,4]] )
A * B                       # elementwise product
#array([[2, 0],
#       [0, 4]])
A @ B                       # matrix product
#array([[5, 4],
#       [3, 4]])
A.dot(B)                    # another matrix product
#array([[5, 4],
#       [3, 4]])



#When operating with arrays of different types, the type of the resulting array 
#corresponds to the more general or precise one (a behavior known as upcasting).
a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3)
b.dtype.name
# 'float64'
c = a+b
c
c.dtype.name
# 'float64'
d = np.exp(c*1j)
d
d.dtype.name
# 'complex128'


a = np.random.random((2,3))
a
#array([[ 0.18626021,  0.34556073,  0.39676747],
#       [ 0.53881673,  0.41919451,  0.6852195 ]])
a.sum()
#2.5718191614547998
a.min()
#0.1862602113776709
a.max()
#0.6852195003967595


b = np.arange(12).reshape(3,4)
b
#array([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11]])

b.sum(axis=0)                            # sum of each column
#array([12, 15, 18, 21])

b.min(axis=1)                            # min of each row
#array([0, 4, 8])

b.cumsum(axis=1)                         # cumulative sum along each row
#array([[ 0,  1,  3,  6],
#       [ 4,  9, 15, 22],
#       [ 8, 17, 27, 38]])



#NumPy provides familiar mathematical functions 
#such as sin, cos, and exp. In NumPy, these are called “universal functions”(ufunc). 
#Within NumPy, these functions operate elementwise on an array, producing an array as output.

B = np.arange(3)
B
#array([0, 1, 2])
np.exp(B)
#array([ 1.        ,  2.71828183,  7.3890561 ])
np.sqrt(B)
#array([ 0.        ,  1.        ,  1.41421356])
C = np.array([2., -1., 4.])
np.add(B, C)
#array([ 2.,  0.,  6.])



import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
#[[ 1.  2.]
# [ 3.  4.]]

a.transpose()
#array([[ 1.,  3.],
#       [ 2.,  4.]])
#
np.linalg.inv(a)
#array([[-2. ,  1. ],
#       [ 1.5, -0.5]])

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
u
#array([[ 1.,  0.],
#       [ 0.,  1.]])
j = np.array([[0.0, -1.0], [1.0, 0.0]])

j @ j        # matrix product
#array([[-1.,  0.],
#       [ 0., -1.]])

np.trace(u)  # trace
#2.0

y = np.array([[5.], [7.]])
np.linalg.solve(a, y)
#array([[-3.],
#       [ 4.]])

np.linalg.eig(j)
#(array([ 0.+1.j,  0.-1.j]), array([[ 0.70710678+0.j        ,  0.70710678-0.j        ],
#       [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]]))