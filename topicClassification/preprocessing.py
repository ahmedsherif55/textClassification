# Import
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import gensim
from bs4 import BeautifulSoup
import time
import numpy as np


class PreProcessing:
    def __init__(self):
        self.data = pd.read_csv('dataset/stack-overflow-data.csv')
        # Extract unique classes
        self.tags = self.data.tags.unique()

    def plot_figures(self):
        plt.figure(figsize=(10, 4))
        self.tags.value_counts().plot(kind='bar')

    def clean_text(self):
        start = time.time()
        # HTML decoding, Default parser is lxml
        self.data['post'] = self.data['post'].apply(lambda text: self.filter_data(text))
        end = time.time()
        print("Time Taken: " + str(end - start))
        #print(self.data['post'][:20])
        return self.data

    def filter_data(self, text):
        # Remove parentheses, brackets and special symbols from input
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,; ]')
        # Don't remove numbers, lower case characters, #, +, _ or space
        # (because they are found in calsses such as: "c++, c#")
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))
        # Transform words to lower case letters
        text = text.lower()
        text = BeautifulSoup(text, 'lxml').get_text()
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        lemmatizer = WordNetLemmatizer()
        # Filter stop words from text
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split(' ') if word not in STOPWORDS)
        # Tokenize will change classes like c#, it will split c and # in two different words
        #text = ' '.join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in STOPWORDS)
        return text

    def word_averaging(self, wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self, wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list])

    def getRawData(self):
        return self.data