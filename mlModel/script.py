# %% [code] {"id":"3JP2pWRJ-Gix","outputId":"a7c7f9da-1b1f-4bad-9e56-c8bdd4c5e456"}
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
nltk.download("stopwords")

# %% [code] {"id":"cBVeNJM9-cpi"}
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# %% [code] {"id":"TuWaYVMQ-eA3"}
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# %% [code] {"id":"eTMFQXoe-nfR"}
downloaded = drive.CreateFile({'id':"1-7T0-xfPTmiAAwcTrOTwusmvp0ajs22u"})   # replace the id with id of file you want to access
downloaded.GetContentFile('train.csv')

# %% [code] {"id":"eqdHEq9V-niz"}
import pandas as pd
tweets=pd.read_csv("train.csv")

# %% [code] {"id":"9f5FUvGx_U5q"}
stop_words=set(stopwords.words("english"))

# %% [code] {"id":"J4ZuRrY-_bOs"}
stop_words.remove("not")

# %% [markdown] {"id":"4KpiK4kFOReF"}
# ###### Function to replace the messaging langugaes and contractions
# 

# %% [code] {"id":"slH0liwdOOvs"}
dickeys=corr_dict.keys()
def conmsgl(text):
        l=text.split()
        for i in l:
            if i.lower() in dickeys:
                i=i.lower()
                k=l.index(i)
                l[k]=corr_dict[i]
        return(" ".join(l))

# %% [code] {"id":"iLOIUJbH_dYN"}
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

# %% [code] {"id":"Bqo-FTN1KStK"}
stp_custom=['i',
            'shan',"ma",'hadn','hasn','doesn','didn','ain','won','me','my','myself', 'we','our','ours','ourselves','you',"you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',        
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don','should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',]

# %% [code] {"id":"YPq0nRT6_eD0","outputId":"dee11cc0-a747-4845-b9a0-c5d6820a0d28"}
text1="Ã°ÂŸÂÂ»Ã°ÂŸÂÂ¸Ã°ÂŸÂÂ¹Ã°ÂŸÂÂŸ #comedians   #cool #igers #igers #instamood"
print("Before Cleaning:",text1)
clean_text1=text_cleaner(text1)
print("After Cleaning: ",clean_text1)
print(len(clean_text1))

# %% [code] {"id":"HeELdljp_glO","outputId":"99591cbe-3fd4-4790-e888-ae90934d8dda"}
cleaned_text=[]
for i in tweets["tweet"]:
    cleaned_text.append(text_cleaner(i))

# %% [code] {"id":"-h5Odn6-LJwA"}


# %% [code] {"id":"9CQDLMAw_iUq","outputId":"5999c7d6-f5df-41dd-aa1d-b56b4cf62e56"}
from ekphrasis.classes.segmenter import Segmenter
seg_tw=Segmenter(corpus="twitter")


# %% [code] {"id":"Gkb49Y6u_pKE"}
def segment(text):
    l=[]
    newstring=text.split()
    for w in newstring:
        l.append(seg_tw.segment(w))
    return(" ".join(l))

# %% [code] {"id":"HMofdjDX_5xm"}
cleaned_tex_seg=[]
for i in cleaned_text:
    cleaned_tex_seg.append(segment(i))

# %% [code] {"id":"p64kozxJ_7hR","outputId":"1770e469-c772-4458-ff9a-0b45c268b032"}
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

# %% [code] {"id":"RuST2as5_9C0"}
def spell_correct(text):
    l=[]
    for i in text.split():
        l.append(sp.correct(i))
    return (" ".join(l))

# %% [code] {"id":"F5ZS5tiW_-ED"}
cleaned_tex_seg_correc=[]
for i in cleaned_tex_seg:
    cleaned_tex_seg_correc.append(spell_correct(i))

# %% [code] {"id":"mUhOPMfUAB2K"}
cleaned_tex_seg_correc1=[]
for i in cleaned_tex_seg_correc:
    if re.findall("xoxo",i):
        i=re.sub("xoxo","hugs kisses",i)
        cleaned_tex_seg_correc1.append(i)
    else:
        cleaned_tex_seg_correc1.append(i)

# %% [code] {"id":"s5srlTUaAB45"}
from sklearn.model_selection import train_test_split
y=tweets["label"]
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(cleaned_tex_seg_correc1,y,test_size=0.3,random_state=0,shuffle=True)

# %% [code] {"id":"sgEd8h3sAUyU","outputId":"1f2ee0d1-31f2-40aa-f122-00a5139961f5"}
import matplotlib.pyplot as plt
text_word_count=[]
for i in cleaned_tex_seg_correc1:
    text_word_count.append(len(i.split()))

length_df=pd.DataFrame({"text":text_word_count})

length_df.hist(bins=5,range=(0,40))
plt.show()

# %% [code] {"id":"tUp39bDZAXr5"}
max_len=16

# %% [code] {"id":"L39ZupqpAZ3l","outputId":"1408aec5-cebd-4743-c7dd-62aad2f66b32"}
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

# %% [code] {"id":"w6DMsZVOAbju"}
from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

# %% [code] {"id":"nVFM79keAdu5","outputId":"847622fb-f6ad-4b56-8947-4d18ca817353"}
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


# %% [code] {"id":"4iF4zpz_AgXX","outputId":"27d3c017-86a1-4226-8cc4-52c7fe4bb2db"}
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
es =EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1, min_delta=1e-7)
mc=ModelCheckpoint("best_model.h5",monitor="val_loss",mode="min",save_best_only=True,verbose=1)
history=model.fit(np.array(x_train),np.array(y_train),batch_size=1000,epochs=50,
                  validation_data=(np.array(x_test),np.array(y_test)),verbose=1,callbacks=[es,mc])

# %% [code] {"id":"whymQ1xGA8sC"}


# %% [code] {"id":"QUHxHbEYA-MQ"}


# %% [code] {"id":"_jU0qE6kBf8r","outputId":"b80b4019-a942-49c4-ca3f-543d99e10dd2"}
from matplotlib import pyplot
pyplot.plot(history.history["loss"],label="train")
pyplot.plot(history.history["val_loss"],label="test")
pyplot.legend()
pyplot.show()

# %% [code] {"id":"cOpbltDkH1XG"}
from keras.models import load_model

# %% [code] {"id":"teH9nzEOH3D4"}
model = load_model('best_model.h5')

# %% [code] {"id":"a7ScF9-7H38q"}
downloaded = drive.CreateFile({'id':"18cgCiM_3KWwbvuFewQkgc-P14CWd7SBb"})   # replace the id with id of file you want to access
downloaded.GetContentFile('test_tweets_anuFYb8.csv')

# %% [code] {"id":"3ow2Ybl7IaTj","outputId":"a4612286-ac99-417c-e580-3df8a0e2ac13"}
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

# %% [code] {"id":"E97SgvqrIkgq"}


# %% [code] {"id":"MxEKt6dhI9jd","outputId":"d1e07250-3801-4179-fc26-7fc475385f28"}
data1["label"]=yhat

data1.to_csv('l1.csv') 
files.download('l1.csv')
