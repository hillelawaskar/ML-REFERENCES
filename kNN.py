# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:56:52 2019

@author: HA5035615
"""

import numpy as np
import operator

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels