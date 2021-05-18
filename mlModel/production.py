import pickle
import re
import os
import numpy as np
import nltk
from keras.models import load_model
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

def preprocess_text(sen):
    # lower the character
    sentence = sen.lower()
    
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    stops = stopwords.words('english')
    
    for word in sentence.split():
        if word in stops:
            sentence = sentence.replace(word, '')
    return sentence


token = None

text = ["Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info."]

with open("token.obj", "rb") as rfile:
    token = pickle.load(rfile)

text = [preprocess_text(x) for x in text]

text = token.texts_to_sequences(text)
text = pad_sequences(text, maxlen=100)


def joiner(file_name):
    paths = os.path.dirname(os.path.abspath(__file__))
    paths = os.path.join(paths, file_name)
    return paths

model = load_model(joiner('model.h5'))

print(np.array(text))
prediction = model.predict(text)

prediction = [0 if y[0] < 0.5 else 1 for y in prediction]

print(prediction)