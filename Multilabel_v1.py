# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:36:40 2019

@author: HA5035615
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
#from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt


tags = pd.read_csv(r'D:\Others\PythonCode\MLinAction\datamulti\Tags.csv')
#queries = pd.read_csv(r'D:\Others\PythonCode\MLinAction\datamulti\Questions.csv', header=0, skiprows=range(1, 999), nrows=1000)
queries = pd.read_csv(r'D:\Others\PythonCode\MLinAction\datamulti\Questions.csv', header=0, nrows=10000)

tags['Tag'].describe()
#count        3750862
#unique         37035   <<< unique tags
#top       javascript
#freq          124155

tags_freq =  tags.groupby('Tag').count()   ## Frequency count 
tags_freq.sort_values('Id', ascending = False, inplace = True, na_position ='last')     # Sorted Tags 

#                           Id
#Tag                          
#javascript             124155
#java                   115212
#c#                     101186
#php                     98808
#android                 90659
#jquery                  78542
#python                  64601


tags['Tag'].loc[np.logical_not (tags['Tag'].isin(['javascript', 'java','C#','php','android','jquery','python' ]))]='others'    # replace other with more than 5 higher frequency items 

#Remove duplicate rows in tags 
tags.drop_duplicates(subset =("Id","Tag"),keep='first',inplace=True)    # first kees the 1st value as non duplicate ; so retatins 1 value 
   
unique_tags = tags.Tag.unique() # contains all unique tags
len(unique_tags)


#Function for removing html tags from the Body of the query 
def removehtml(html):
  recmpl = re.compile('<.*?>')
  text = re.sub(recmpl, '', html)
  return text       

#queries['Body'][0]
##aaaa = removehtml(queries['Body'][0])   to remove html tags 

queries['question'] = queries['Body'].apply(removehtml)      # remove html tags for all the df column body
queries['joinquestion'] = queries [['Title', 'question']].apply(lambda x : ''.join(x),axis=1 )  # join title and the body 
                   
#create new df for combining the Title body and labesl with one hot encoded 
df=pd.DataFrame(columns=['Id'])   # create df with Id
df['joinquestion'] = None   # add another column to df 
df = pd.concat([df,pd.DataFrame(columns = unique_tags)])  # add tags as column to df 

              

# PD Merge based on ID and do the one hot encoding as well              
              
flag = 0            
indx = 0

for query in queries.iterrows():
    if flag != 0 :
        indx += 1
    else :
        flag = 1
    print (query[1][0])
    
    #print (query[1][5])   
    df.ix[str(indx),'Id']= query[1][0]   # Id
    df.ix[str(indx),'joinquestion']= str(query[1][8])  # Title
    tag_sub = tags[tags['Id']==query[1][0]]  
    for tag_su in tag_sub.iterrows():
        #print (tag_su[1][1])   # this is individual tag
        df.ix[str(indx),str(tag_su[1][1])]= 1
              

#Save the data 
df.to_csv(r'D:\Others\PythonCode\MLinAction\datamulti\tagged_data10000.csv')      # save df         
#df = pd.read_csv(r'D:\Others\PythonCode\MLinAction\datamulti\tagged_data10000.csv',encoding="ISO-8859-1",  index_col=[0])

# No Need to run below statement now as we have transformed the data to contain others 
df.dropna(axis = 1, how ='all', inplace = True)    # remove all columns that have Nan in all rows i.e. the tags are not used 

         
#seprate X and Y
X_df = df['joinquestion']
Y_df = df.drop(['Id','joinquestion'], axis = 1)
Y_df.fillna(0, inplace=True)
Y_df.shape


X_train,X_test,y_train,y_test = train_test_split(X_df,Y_df,test_size = 0.2)

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


# Learn to predict each class against the other
classifier = Pipeline([('vectorizer', CountVectorizer(stop_words = stopWords)),('tfidf', TfidfTransformer()),('clf' , OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced')))])
#classifier = Pipeline([('vectorizer', CountVectorizer(stop_words = stopWords)),('tfidf', TfidfTransformer()),('clf' , OneVsRestClassifier(SVC(kernel='poly', probability=True, class_weight='balanced')))])
#classifier = Pipeline([('vectorizer', CountVectorizer(stop_words = stopWords)),('tfidf', TfidfTransformer()),('clf' , OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto')))])


classifier.fit(X_train, y_train)
proba = classifier.predict_proba(X_test)   # to find probability 
 
predicted = classifier.predict(X_test)



#Measuring metrics 

#y_score = classifier.decision_function(X_test)
#from sklearn.metrics import average_precision_score
#average_precision = average_precision_score(y_test, y_score)
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))
#
#from sklearn.metrics import f1_score
#f1_score_ =  f1_score(y_test, predicted, average = 'macro') 


# For each class
from sklearn.metrics import classification_report
report = classification_report(y_test, predicted)
print(report)

from sklearn.metrics import multilabel_confusion_matrix
conf_report = multilabel_confusion_matrix(y_test, predicted)

print(type(conf_report))

labels = ['android',  'java' , 'javascript',  'jquery',  'others' , 'php' , 'python']
count = 0
for i in conf_report:
    print (labels[count])
    print (i)
    count += 1




predicted_1 = classifier.predict(('I need to do and C++ Java and Python with air',)) 
predicted_1 

print ("Accuracy Score: ",accuracy_score(y_test, predicted)) 


# print the outputed tags for y_test 
for i in y_test.iterrows():
    #print (type(i[1]))
    pred_lab = ""
    for (column, Data) in i[1].iteritems():
        if Data == 1:
            pred_lab = pred_lab + ',' + column
    print (str(i[0]) + ' ' + pred_lab  )
    
    
    
