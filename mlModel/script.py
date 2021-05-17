import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
nltk.download("stopwords")

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

downloaded = drive.CreateFile({'id':"1-7T0-xfPTmiAAwcTrOTwusmvp0ajs22u"})   # replace the id with id of file you want to access
downloaded.GetContentFile('train.csv')

import pandas as pd
tweets=pd.read_csv("train.csv")

stop_words=set(stopwords.words("english"))

stop_words.remove("not")

dickeys=corr_dict.keys()
def conmsgl(text):
        l=text.split()
        for i in l:
            if i.lower() in dickeys:
                i=i.lower()
                k=l.index(i)
                l[k]=corr_dict[i]
        return(" ".join(l))

def text_cleaner(text):
    #converting to lowercase
    newstring=text.lower()
     #fetching all the alphabetic characters
    newstring=re.sub("[^a-zA-Z]"," ",newstring)
    newstring=re.sub("o m a y g a d","oh my god",newstring)
    #removing words inside ()
    newstring=re.sub(r'\([^)]*\)',"",newstring)
    #removing words inside {}
    newstring=re.sub(r'{[^)]*\}',"",newstring)
    #removing words inside[]
    newstring=re.sub(r'\[[^)]*\]',"",newstring)
    #removing stopwords
    tokens=[w for w in newstring.split() if not w in stp_custom]
    longwords=[]
    #longwords=[w for w in newstring.split() if len(w)>=4]
    print(longwords)
    for i in tokens:
       #removing short words
        if len(i)>=3:
            longwords.append(i)
            
    return(" ".join(longwords)).strip()

stp_custom = [
    "i",
    "shan",
    "ma",
    "hadn",
    "hasn",
    "doesn",
    "didn",
    "ain",
    "won",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
]

text1="Ã°ÂŸÂÂ»Ã°ÂŸÂÂ¸Ã°ÂŸÂÂ¹Ã°ÂŸÂÂŸ #comedians   #cool #igers #igers #instamood"
print("Before Cleaning:",text1)
clean_text1=text_cleaner(text1)
print("After Cleaning: ",clean_text1)
print(len(clean_text1))

cleaned_text=[]
for i in tweets["tweet"]:
    cleaned_text.append(text_cleaner(i))

from ekphrasis.classes.segmenter import Segmenter
seg_tw=Segmenter(corpus="twitter")


def segment(text):
    l=[]
    newstring=text.split()
    for w in newstring:
        l.append(seg_tw.segment(w))
    return(" ".join(l))

cleaned_tex_seg=[]
for i in cleaned_text:
    cleaned_tex_seg.append(segment(i))

from ekphrasis.classes.spellcorrect import SpellCorrector
sp = SpellCorrector(corpus="english")
def spell_correct(text):
    l=[]
    for i in text.split():
        l.append(sp.correct(i))
    return (" ".join(l))
cleaned_tex_seg_correc=[]
for i in cleaned_tex_seg:
    cleaned_tex_seg_correc.append(spell_correct(i))

def spell_correct(text):
    l=[]
    for i in text.split():
        l.append(sp.correct(i))
    return (" ".join(l))

cleaned_tex_seg_correc=[]
for i in cleaned_tex_seg:
    cleaned_tex_seg_correc.append(spell_correct(i))

cleaned_tex_seg_correc1=[]
for i in cleaned_tex_seg_correc:
    if re.findall("xoxo",i):
        i=re.sub("xoxo","hugs kisses",i)
        cleaned_tex_seg_correc1.append(i)
    else:
        cleaned_tex_seg_correc1.append(i)

from sklearn.model_selection import train_test_split
y=tweets["label"]
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(cleaned_tex_seg_correc1,y,test_size=0.3,random_state=0,shuffle=True)

import matplotlib.pyplot as plt
text_word_count=[]
for i in cleaned_tex_seg_correc1:
    text_word_count.append(len(i.split()))

length_df=pd.DataFrame({"text":text_word_count})

length_df.hist(bins=5,range=(0,40))
plt.show()

max_len=16

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer()
###Creating index for a word
tokenizer.fit_on_texts(list(x_train))

#converting word seq to integer seq
x_train=tokenizer.texts_to_sequences((x_train))
x_test=tokenizer.texts_to_sequences((x_test))

#padding with zero
x_train=pad_sequences(x_train,maxlen=max_len,padding="post")
x_test=pad_sequences(x_test,maxlen=max_len,padding="post")

vocabulary=len(tokenizer.word_index)+1
print("Vocabulary Size:",vocabulary)

from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras.backend as K
from keras.layers import SpatialDropout1D
K.clear_session()
model=Sequential()
model.add(Embedding(vocabulary,5,input_length=max_len,trainable=True,mask_zero=True))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(3,dropout=0.1,recurrent_dropout=0.2))
model.add(Dense(2,activation="relu"))
model.add(Dense(2,activation="sigmoid"))
print(model.summary())


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
es =EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1, min_delta=1e-7)
mc=ModelCheckpoint("best_model.h5",monitor="val_loss",mode="min",save_best_only=True,verbose=1)
history=model.fit(np.array(x_train),np.array(y_train),batch_size=1000,epochs=50,
                  validation_data=(np.array(x_test),np.array(y_test)),verbose=1,callbacks=[es,mc])





from matplotlib import pyplot
pyplot.plot(history.history["loss"],label="train")
pyplot.plot(history.history["val_loss"],label="test")
pyplot.legend()
pyplot.show()

from keras.models import load_model

model = load_model('best_model.h5')

downloaded = drive.CreateFile({'id':"18cgCiM_3KWwbvuFewQkgc-P14CWd7SBb"})   # replace the id with id of file you want to access
downloaded.GetContentFile('test_tweets_anuFYb8.csv')

model = load_model('best_model.h5')
data1=pd.read_csv("test_tweets_anuFYb8.csv")
cleaned_text_test=[]
for i in data1["tweet"]:
    cleaned_text_test.append(text_cleaner(i))
def segment(text):
    l=[]
    newstring=text.split()
    for w in newstring:
        l.append(seg_tw.segment(w))
    return(" ".join(l))
cleaned_tex_test_seg=[]
for i in cleaned_text_test:
    cleaned_tex_test_seg.append(segment(i))
cleaned_tex_seg_correc_test=[]
for i in cleaned_tex_test_seg:
    cleaned_tex_seg_correc_test.append(spell_correct(i))
cleaned_tex_seg_correc1_test=[]
for i in cleaned_tex_seg_correc_test:
    if re.findall("xoxo",i):
        i=re.sub("xoxo","hugs kisses",i)
        cleaned_tex_seg_correc1_test.append(i)
    else:
        cleaned_tex_seg_correc1_test.append(i)
x_val=tokenizer.texts_to_sequences(cleaned_tex_seg_correc1_test)
x_val=pad_sequences(x_val,maxlen=max_len,padding="post")
yhat = model.predict_classes(x_val)

data1["label"]=yhat

data1.to_csv('l1.csv') 
files.download('l1.csv')

