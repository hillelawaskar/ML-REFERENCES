# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:50:38 2019


"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from scipy import stats
#For some Statistics
from scipy.stats import norm, skew
from sklearn.preprocessing import Imputer


df = pd.read_csv(r'D:\Others\PythonCode\MLinAction\Feature engg data\data.csv',index_col=0)

df.head()
shape(df)

df.describe()
print (df.items)

df['SalePrice'].describe()
# Handling Outliers

#Scatter plot GrLivArea vs SalePrice
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
                 

#Scatter plot TotalBsmtSF vs SalePrice

data = pd.concat([df['SalePrice'], df['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000));    
                 
                 
                 
# Box plot of OverallQual vs SalePrice

data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
        
        
        
#Deleting outliers
df = df.drop(df[(df['GrLivArea']>3000) & (df['GrLivArea']<6000)].index)

#Check the graph again
fig, ax = plt.subplots()
ax.scatter(df['GrLivArea'], df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()




# missing data
total = df.isnull().sum(axis=1).sort_values(ascending=False)
percent = (df.isnull().sum(axis=1)/df.isnull().count(axis=1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total_missing_values_per_row', 'Percent'])
missing_data.head(20)


# mark the missing data 
missing_data.Total_missing_values_per_row.nonzero()

# remove rows that contain missing values 
df_copy = df.copy().dropna(how='any')
df_copy.shape


#A slightly better approach would be to drop only those rows, which had all the values as null. This would surely work better than the previous approach.
#Let's check the number of houses, with at least some data. 
df_copy = df.copy().dropna(how='all')
df.shape


#missing data observing in columns
total = df.isnull().sum(axis=0).sort_values(ascending=False)
percent = ((df.isnull().sum(axis=0)/df.isnull().count(axis=0))*100).sort_values(ascending=False)

# count the number of null values in the column and their perecentage of the total data
missing_data_columns = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_columns.head(20)



# droping columns containing high missing values 
df1=df.copy()
df1.drop(["PoolQC", "MiscFeature","Alley","Fence"], axis = 1, inplace = True)

# display the columns left after droping the above features
list(df1.columns.values)[:20]




#histogram and normal probability plot
sns.distplot(df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)


skewed_grLiv = skew(df['GrLivArea'])
print(skewed_grLiv)
