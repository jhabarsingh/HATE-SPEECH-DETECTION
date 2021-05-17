import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def process(text):
    text = text.lower()
    
    text = text.replace('[^\s\w]','')
    text = text.replace('[^\s\w]','')
    
    nltk.download('punkt')
    text = word_tokenize(text)

    ps = PorterStemmer() 
    text = list(ps.stem(i) for i in text)

    nltk.download('stopwords')
    stops = set(stopwords.words("english"))
    text = ' '.join(list(i for i in text if i not in stops))
    

    nltk.download('wordnet')
    lamatizer = WordNetLemmatizer()
    text = lamatizer.lemmatize(text)

    tokenizer = Tokenizer(num_words = 4500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = text)
    X1 = tokenizer.texts_to_sequences(texts = text)
    word_index = tokenizer.word_index

    data1 = pad_sequences(sequences= X1 , maxlen = 1000)
    class1 = to_categorical(np.asarray(df1['class']), num_classes = 3)
    return [data1, class1]



if __name__ == "__main__":
    text = "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out..."
    str = process(text)

    print(str)