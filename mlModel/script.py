from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, concatenate
from keras.models import Model, Sequential
from tensorflow.keras import regularizers

for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df1 = pd.read_csv('./dataset/labeled_data.csv')
del df1[df1.columns[0]]

# Data Pre-Processing

# Converting all string to lower case
df1 = df1.apply(lambda x: x.astype(str).str.lower())

# Removing Punctuations
df1.tweet = df1.tweet.str.replace('[^\s\w]','')

# Removing HTML Tags
df1.tweet = df1.tweet.str.replace('[^\s\w]','')

# Tokenizing
nltk.download('punkt')
df1['tweet_token'] = df1['tweet'].apply(lambda x: word_tokenize(x))

# Stemming
ps = PorterStemmer() 
df1.tweet = df1.tweet_token.apply(lambda x: list(ps.stem(i) for i in x))

# Removing the stop words and Rejoining 
nltk.download('stopwords')
stops = set(stopwords.words("english"))                  
df1.tweet = df1.tweet.apply(lambda x: ' '.join(list(i for i in x if i not in stops)))

# Lammatizing
nltk.download('wordnet')
lamatizer = WordNetLemmatizer()
df1.tweet.apply(lambda x: lamatizer.lemmatize(x))


tokenizer = Tokenizer(num_words = 4500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts = df1.tweet)
X1 = tokenizer.texts_to_sequences(texts = df1.hate_speech)
X2 = tokenizer.texts_to_sequences(texts = df1.offensive_language)
X3 = tokenizer.texts_to_sequences(texts = df1.neither)
X4 = tokenizer.texts_to_sequences(texts = df1.tweet)
word_index = tokenizer.word_index

data1 = pad_sequences(sequences= X1 , maxlen = 1000)
class1 = to_categorical(np.asarray(df1['class']), num_classes = 3)

data2 = pad_sequences(sequences= X2 , maxlen = 1000)
class2 = to_categorical(np.asarray(df1['class']), num_classes = 3)

data3 = pad_sequences(sequences= X3 , maxlen = 1000)
class3 = to_categorical(np.asarray(df1['class']), num_classes = 3)

data4 = pad_sequences(sequences= X4 , maxlen = 1000)
class4 = to_categorical(np.asarray(df1['class']), num_classes = 3)

"""
print('Length of data1 tensor:', data1.shape)
print('Length of labels1 tensor:', class1.shape)
print('Length of data1 tensor:', data2.shape)
print('Length of labels1 tensor:', class2.shape)
print('Length of data1 tensor:', data3.shape)
print('Length of labels1 tensor:', class3.shape)
print('Length of data1 tensor:', data4.shape)
print('Length of labels1 tensor:', class4.shape)
"""

indices1 = np.arange(df1.shape[0])
np.random.shuffle(indices1)
data1 = data1[indices1]
class1 = class1[indices1]
x_train1, x_test1, y_train1, y_test1 = train_test_split(data1, class1, test_size=0.2, random_state=42)
x_test1, x_val1, y_test1, y_val1 = train_test_split(data1, class1, test_size=0.4, random_state=42)

"""
print('1:')
print(x_train1.shape)
print(y_train1.shape)
print(x_test1.shape)
print(y_test1.shape)
print(x_val1.shape)
print(y_val1.shape)
"""


#Using Pre-trained word embeddings
MAX_SEQUENCE_LENGTH = 1000
GLOVE_DIR = "./dataset/" 
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=1000)

# LSTM MODEL
### Model for 1st data set (test_data1)
def LSTM_model(n, x_train, y_train, x_val, y_val, x_test, y_test):
    lstm_model = Sequential()
    lstm_model.add(Embedding(len(word_index) + 1, 100, weights = [embedding_matrix], input_length = MAX_SEQUENCE_LENGTH, trainable = False))
    lstm_model.add(LSTM(128))
    lstm_model.add(Dense(128, activation = 'relu'))
    lstm_model.add(Dense(64, activation = 'relu'))
    lstm_model.add(Dense(32, activation = 'relu'))
    lstm_model.add(Dense(3, activation = 'softmax'))
    lstm_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    lstm_model.summary()
    
    # GENERATE BIN FILE
    lstm_model.save('pickle.h5')

    history = lstm_model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 5, batch_size = 128)
    print('\nModel Training Complete !')
    
    ### PREDICTION
    y_preds = lstm_model.predict(x_test)
    y_pred = np.round(y_preds)
    cpred = float(sum(y_pred == y_test)[0])
    cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
    print("\n-> Correct predictions:", cpred)
    print("\n-> Total number of test examples:", len(y_test))
    print("\n-> Accuracy of model: ", cpred/float(len(y_test)))
    print("\n-> Confusion for Dataset",n,": ", cm)

    return history

history = LSTM_model(1, x_train1, y_train1, x_val1, y_val1, x_test1, y_test1)

# list all data in history

print(history.history.keys())

print("ACCURACY\n")
print(history.history['accuracy'])
print(history.history['val_accuracy'])

print("LOSS\n")
print(history.history['loss'])
print(history.history['val_loss'])

