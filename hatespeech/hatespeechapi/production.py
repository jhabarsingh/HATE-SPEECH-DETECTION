import pickle
import re
import os
import numpy as np
import nltk
from keras.models import load_model
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence

def joiner(file_name):
        paths = os.path.dirname(os.path.abspath(__file__))
        paths = os.path.join(paths, file_name)
        return paths

def predict(text):
    x_test = [text]
    
    x_tokenizer = None
    with open(joiner("toxic_tokenizer.pkl"), "rb") as rfile:
        x_tokenizer = pickle.load(rfile)

    max_text_length = 400

    model = load_model(joiner('toxic_cnn.h5'))

    x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
    x_testing = sequence.pad_sequences(x_test_tokenized,maxlen=max_text_length)

    y_pred = model.predict(x_testing,verbose=1,batch_size=32)

    y_pred = [0 if y[0] < 0.5 else 1 for y in y_pred]
    return y_pred[0]


