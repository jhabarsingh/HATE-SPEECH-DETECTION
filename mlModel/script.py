from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df1 = pd.read_csv('/kaggle/input/hate-speech-and-offensive-language-dataset/labeled_data.csv')

print(df1.columns)
df1.head(10)

del df1[df1.columns[0]]

# # Data Pre-Processing

df1.isnull().sum()

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

df1.columns

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

print('Length of data1 tensor:', data1.shape)
print('Length of labels1 tensor:', class1.shape)
print('Length of data1 tensor:', data2.shape)
print('Length of labels1 tensor:', class2.shape)
print('Length of data1 tensor:', data3.shape)
print('Length of labels1 tensor:', class3.shape)
print('Length of data1 tensor:', data4.shape)
print('Length of labels1 tensor:', class4.shape)

indices1 = np.arange(df1.shape[0])
np.random.shuffle(indices1)
data1 = data1[indices1]
class1 = class1[indices1]
x_train1, x_test1, y_train1, y_test1 = train_test_split(data1, class1, test_size=0.2, random_state=42)
x_test1, x_val1, y_test1, y_val1 = train_test_split(data1, class1, test_size=0.4, random_state=42)


print('1:')
print(x_train1.shape)
print(y_train1.shape)
print(x_test1.shape)
print(y_test1.shape)
print(x_val1.shape)
print(y_val1.shape)

#Using Pre-trained word embeddings
MAX_SEQUENCE_LENGTH = 1000
GLOVE_DIR = "../input/glove-global-vectors-for-word-representation/" 
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

# # Deep CNN

### Model for 1st data set (test_data1)
def DCNN_model(n, x_train, y_train, x_val, y_val, x_test, y_test):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1= Conv1D(8, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(8, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_cov3 = Conv1D(8, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(8, activation='relu')(l_flat)
    l_dense1 = Dense(8, activation='relu')(l_dense)
    l_dense2 = Dense(8, activation='relu')(l_dense1)
    preds = Dense(3, activation='softmax')(l_dense2)

    dcnn_model = Model(sequence_input, preds)
    dcnn_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['acc'])
    print("Fitting the simple convolutional neural network model")
    dcnn_model.summary()
    history = dcnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=10, batch_size=8)
    print('\nModel Training Completed !')

    ### PREDICTING 
    y_preds = dcnn_model.predict(x_test)
    y_pred = np.round(y_preds)
    cpred = float(sum(y_pred == y_test)[0])
    cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
    print("\n-> Correct predictions:", cpred)
    print("\n-> Total number of test examples:", len(y_test))
    print("\n-> Accuracy of model: ", cpred/float(len(y_test)))
    print("\n-> Confusion matrix for Dataset",n,": ", cm)

    plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
    plt.title('Confusion matrix - CNN Model 1')
    plt.colorbar()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    plt.show()
    return history


# %% [code] {"jupyter":{"outputs_hidden":true}}
history = DCNN_model(x_train1, y_train1, x_val1, y_val1, x_test1, y_test1,
                     x_train2, y_train2, x_val2, y_val2, x_test2, y_test2,
                     x_train3, y_train3, x_val3, y_val3, x_test3, y_test3,
                     x_train4, y_train4, x_val4, y_val4, x_test4, y_test4)
print(' ')

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# # Multi Channel CNN

def define_model(x_train1, y_train1, x_val1, y_val1, x_test1, y_test1):
    
    sequence_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences1 = embedding_layer(sequence_input1)
    cov1= Conv1D(32, 5, activation='relu')(embedded_sequences1)
    pool1 = MaxPooling1D(5)(cov1)
    flat1 = Flatten()(pool1)

    sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences2 = embedding_layer(sequence_input2)
    cov2 = Conv1D(32, 5, activation='relu')(embedded_sequences2)
    pool2 = MaxPooling1D(5)(cov2)
    flat2 = Flatten()(pool2)

    sequence_input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences3 = embedding_layer(sequence_input3)
    cov3 = Conv1D(32, 5, activation='relu')(embedded_sequences3)
    pool3 = MaxPooling1D(35)(cov3)
    flat3 = Flatten()(pool3)
    
#     sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences4 = embedding_layer(sequence_input4)
#     cov4 = Conv1D(32, 5, activation='relu')(embedded_sequences4)
#     pool4 = MaxPooling1D(35)(cov4)
#     flat4 = Flatten()(pool4)

    merge = concatenate([flat1, flat2, flat3])

    # flat4 = Flatten()(merge)
    dense = Dense(32, activation='relu')(merge)
    preds = Dense(3, activation='softmax')(dense)

    model = Model(inputs = [sequence_input1, sequence_input2, sequence_input3], outputs = preds)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['acc'])
    print("Fitting the simple convolutional neural network model")
    model.summary()
    # history = model.fit([x_train, x_train, x_train], y_train, epochs=2, batch_size=32)
    history = model.fit([x_train1, x_train2, x_train3], y_train1,validation_data = ([x_val1, x_val1, x_val1], y_val1), epochs=10, batch_size=32)
    print('\nModel Training Completed !')

    ### PREDICTION
    y_pred = np.round(model.predict([x_test1, x_test1, x_test1]))
    cpred = float(sum(y_pred == y_test1)[0])
    cm = confusion_matrix(y_test1.argmax(1), y_pred.argmax(1))
    print("\n-> Correct predictions:", cpred)
    print("\n-> Total number of test examples:", len(y_test1))
    print("\n-> Accuracy of model: ", cpred/float(len(y_test1)))
    print("\n-> Confusion Matrix: ", cm)

    plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
    plt.title('Confusion matrix - CNN Model 1')
    plt.colorbar()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return history

# %% [code]
history = define_model(x_train1, y_train1, x_val1, y_val1, x_test1, y_test1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# # LSTM

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
    history = lstm_model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 15, batch_size = 128)
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

    plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
    plt.title('Confusion matrix - CNN Model 1')
    plt.colorbar()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    plt.show()
    return history

history = LSTM_model(1, x_train1, y_train1, x_val1, y_val1, x_test1, y_test1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

