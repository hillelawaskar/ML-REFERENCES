# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:39:43 2019

@author: 
"""
############# to find specific string in one column in excel ; comparted to other 

import re
import pandas as pd

data = pd.read_csv(r'D:\Others\PythonCode\MLinAction\Re data\data.csv')
data1 = data
data1['add'] = None

for ix,dat in data1.iterrows():
    for ixa,dataa in data.iterrows():
        my_regex = dat['c']
        a = re.search(my_regex, dataa[0], re.IGNORECASE)
        if a != None or data1['add'] is None:
            data1['add'][ix] = dataa[0] 
            break
print( data1)
