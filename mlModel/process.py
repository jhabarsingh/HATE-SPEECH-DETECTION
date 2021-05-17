import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def processData(text):
    df1 = pd.read_csv("./dataset/labeled_data.csv")
    del df1[df1.columns[0]]
    df1.tweet[0] = text
    # Data Pre-Processing
    print(df1.tweet[0])
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
    
    return [data1[0]]




if __name__ == "__main__":
    text = " Keeks is a bitch she curves everyone "" lol I walked into a conversation like this. Smh"
    str = processData(text)

    print(str)