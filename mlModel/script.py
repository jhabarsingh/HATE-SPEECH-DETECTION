import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(filename)

# Load data
train_df = pd.read_csv('./dataset/toxic_train.csv')
train_df.head()

train_df.sample(10,random_state=1)

x = train_df['comment_text']
y = train_df['toxic']

# View some toxic comments
train_df[train_df.toxic==1].sample(5)

comments = train_df['comment_text'].loc[train_df['toxic']==1].values


train_df['toxic'].value_counts()


max_features = 20000
max_text_length = 400

x_tokenizer = text.Tokenizer(max_features)
x_tokenizer.fit_on_texts(list(x))

# Save tokenizer for future use
with open('toxic_tokenizer.pkl', 'wb') as f:
    pickle.dump(x_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

x_tokenized = x_tokenizer.texts_to_sequences(x)
x_train_val= sequence.pad_sequences(x_tokenized, maxlen=max_text_length)

embedding_dim =100
embeddings_index = dict()
f = open('./dataset/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word]= coefs
f.close()
print(f'Found {len(embeddings_index)} word vectors')

embedding_matrix= np.zeros((max_features,embedding_dim))
for word, index in x_tokenizer.word_index.items():
    if index>max_features-1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index]= embedding_vector

# # Building Model

filters= 250
kernel_size=3
hidden_dims= 250

model = Sequential()
model.add(Embedding(max_features,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False)
        )

model.add(Dropout(0.2))

model.add(Conv1D(filters,
            kernel_size,
            padding='valid',
            activation='relu')
        )

model.add(MaxPooling1D())

model.add(Conv1D(filters,
            5,
            padding='valid',
            activation='relu')
        )

model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
model.summary()

# # Compiling the Model

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

x_train,x_val,y_train,y_val = train_test_split(x_train_val,y,test_size=0.3,random_state=1)

batch_size= 64
epochs = 3
hist = model.fit(x_train,y_train,
                batch_size= batch_size,
                epochs=epochs,
                validation_data= (x_val,y_val)
            )


model.save('toxic_cnn.h5')

# test_df = pd.read_csv('./dataset/toxic_test.csv')

# x_test = test_df['comment_text'].values
# y_test = test_df['toxic'].values

x_test = ["hi how are you"]
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
x_testing = sequence.pad_sequences(x_test_tokenized,maxlen=max_text_length)

y_pred = model.predict(x_testing,verbose=1,batch_size=32)

y_pred = [0 if y[0] < 0.5 else 1 for y in y_pred]
print(y_pred)

x_test = ["Get the fuck out of here"]
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
x_testing = sequence.pad_sequences(x_test_tokenized,maxlen=max_text_length)

y_pred = model.predict(x_testing,verbose=1,batch_size=32)

y_pred = [0 if y[0] < 0.5 else 1 for y in y_pred]
print(y_pred)
