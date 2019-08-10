# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:49:02 2019


"""

#1) Create Data 
import numpy as np 
x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)

data = 2 * np.random.random((10, 10)) 
data2 = 3 * np.random.random((10, 10)) 
Y, X = np.mgrid[-3:3:100j, -3:3:100j] 
U = -1 - X**2 + Y 
V = 1 + X - Y**2 
from matplotlib.cbook import get_sample_data 
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))



#2) Create Plot 
import matplotlib.pyplot as plt
#2.a ) Figure 
fig = plt.figure() 
ax = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))
 
#2.b Axes All plotting is done with respect to an Axes. In most cases, a subplot will fit your needs. A subplot is an axes on a grid system.


#3) Plotting Unit s
plt.plot(x,y)
plt.plot(x,z)
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.
plt.show()


plt.scatter(x,y)  



#4) Customize plot 



#5) Save Plt 


#6) Show plot 


###########LEARN MATPLOT LIB 


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")
plt.plot(x,y)



#### Add Figure size 

x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)
plt.figure(figsize = (10,20) )  # x until 10 with 20 bins(or parts) each
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")
plt.plot(x,y)


x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)
plt.figure(figsize = (10,20) )  # x until 10 with 20 bins(or parts) each
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")
plt.plot(x,y,"go")     # green dots 

        

x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)
#plt.figure(figsize = (10,20) )  # x until 10 with 20 bins(or parts) each
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")
plt.plot(x,y,'go',x,z ,'r^')     # green dots 

   ￼     

 x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)       
plt.subplot(1,2,1)
plt.plot(x,y,'go')
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.subplot(1,2,2)
plt.plot(x,z ,'r^')
#plt.figure(figsize = (10,20) )  # x until 10 with 20 bins(or parts) each
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")



x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)       
plt.subplot(2,1,1)
plt.plot(x,y,'go')
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.subplot(2,1,2)
plt.plot(x,z ,'r^')
#plt.figure(figsize = (10,20) )  # x until 10 with 20 bins(or parts) each
plt.xlabel("X hai ye")
plt.ylabel("Y hai ye")
plt.title("Fundu graph")
           

# Now using subplots 

x = np.linspace(0, 10, 100) 
y = np.cos(x) 
z = np.sin(x)       
fig,ax = plt.subplots(nrows=2, ncols=2 , figsize = (10,10))
ax[0,0].plot(x,y,'go')
ax[1,1].plot(x,z ,'r^')


##BAR graphs 
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))   ## convert str into numbers and then use xticks
performance = [10,8,6,4,2,1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()


########


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]
variance = [0.2,0.1,0.4,0.3,0.2,0.2]


plt.barh(y_pos, performance, xerr= variance,align='center', alpha=0.5)   ## added  variance too 
plt.yticks(y_pos, objects)
plt.xlabel('Usage')
plt.title('Programming language usage')

plt.show()



############## 2 BAR S


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
means_frank = (90, 55, 40, 65)
means_guido = (85, 62, 54, 20)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,alpha=opacity,color='b',label='Frank')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,alpha=opacity,color='g',label='Guido')

plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.legend()

plt.tight_layout()
plt.show()


############STACKED BAR 


fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,alpha=opacity,color='b',label='Frank')

rects2 = plt.bar(index , means_guido, bar_width,alpha=opacity,color='g',label='Guido', bottom = means_frank)

plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.legend()

plt.tight_layout()
plt.show()




#####Pie Charts 



import matplotlib.pyplot as plt

# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


###############

import matplotlib.pyplot as plt

labels = ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake']
sizes = [38.4, 40.6, 20.7, 10.3]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()





########## histogram 

import matplotlib.pyplot as plt
 
x = [1,1,2,3,3,5,7,8,9,10,
     10,11,11,13,13,15,16,17,18,18,
     18,19,20,21,21,23,24,24,25,25,
     25,25,26,26,26,27,27,27,27,27,
     29,30,30,31,33,34,34,34,35,36,
     36,37,37,38,38,39,40,41,41,42,
     43,44,45,45,46,47,48,48,49,50,
     51,52,53,54,55,55,56,57,58,60,
     61,63,64,65,66,68,70,71,72,74,
     75,77,81,83,84,87,89,90,90,91
     ]

plt.hist(x, bins=10)



########### SCATTER PLOT 

import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


#  3 D plots 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()


