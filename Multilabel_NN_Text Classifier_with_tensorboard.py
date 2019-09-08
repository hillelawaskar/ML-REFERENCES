# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:35:19 2019

@author: hillel

"""
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
import pandas as pd
import matplotlib.pyplot as plt


####for tensor board start
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
####for tensor board end

df = pd.read_csv(r'D:\Data\MLinAction\MultiLabel\tagged_data20000.csv',encoding="ISO-8859-1",  index_col=[0])

#seprate X and Y
X_df = df['joinquestion']
Y_df = df.drop(['Id','joinquestion'], axis = 1)
Y_df.fillna(0, inplace=True)
Y_df.shape

Y_df.sum(axis=0).plot.bar()

X_train,X_test,y_train,y_test = train_test_split(X_df,Y_df,test_size = 0.2)


tokenizer = Tokenizer(num_words=5000)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


X1_train = tokenizer.texts_to_sequences(X_train)
X1_test = tokenizer.texts_to_sequences(X_test)
print('word_index : ',tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 500

X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)


input_1 = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100)(input_1)
LSTM_Layer1 = LSTM(128)(embedding_layer)

output = Dense(7, activation='sigmoid')(LSTM_Layer1)
model = Model(inputs=input_1, outputs=output)

####for tensor board start
tensorboard = TensorBoard(log_dir = '/logs/{}'.format(time()))
####for tensor board end

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

####for tensor board start
history = model.fit(x=X1_train, y=y_train, batch_size=20, epochs=5, verbose=1, validation_split=0.2, callbacks = [tensorboard])
####for tensor board end

score = model.evaluate(x=X1_test, y=y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()