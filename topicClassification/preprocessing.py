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

    def clean_text(self):
        """  Clean dataset to reduce noise.

            This function applies a filter function to data records.

            Args:
                None.

            Returns:
                pandas dataframe: Filtered Dataframe

            Raises:
                None.

            Examples:
                None.
        """
        start = time.time()
        self.data['post'] = self.data['post'].apply(lambda text: self.filter_data(text))
        end = time.time()
        print("Pre-processing Time Taken: " + str(end - start))
        return self.data

    def filter_data(self, text):
        """  Filter and clean data record.

            This function filters a record (string) by applying two regular expressions, one to
            remove some special symbols like brackets or parentheses and the other one removes
            some words that begin with specials symbols, then lower case all letters and filter them
            from stopwords then finally lemmatize words in the string and return it.

            Args:
                None.

            Returns:
                str: Filtered string.

            Raises:
                None.

            Examples:
                None.
        """
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

    def word_averaging(self, glove_model, words):
        """  Calculate Mean word embedding for sentence.

            This function calculate mean word embedding for sentence by acquiring embedding
            for each word in sentence then extract normalized word vector and finally scale
            the mean list of vectors to a scalar value.

            Args:
                glove_model:
                words:

            Returns:
                str: Filtered string.

            Raises:
                None.

            Examples:
                None.
        """
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in glove_model.vocab:
                # model.syn0norm is matrix, that contains normalized word-vectors
                # Retrieve normalized word vector for current word and append to mean
                mean.append(glove_model.syn0norm[glove_model.vocab[word].index])
                # Append word itself to all_words
                all_words.add(glove_model.vocab[word].index)

        if not mean:
            # If not words are found, mean si initialized with zero
            return np.zeros(glove_model.vector_size, )

        # Scale vector to unit length (scalar value) with type float
        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self, glove_model, text_list):
        """  Applies word_averaging function to each item in dataset

            Applies word_averaging function to each item in dataset then stack them into
            one-dimensional numpy array

            Args:
                glove_model (word2vec): glove Pre-trained Embedding
                text_list (list): list of posts (strings)

            Returns:
                numpy: one dimensional array (flattened).

            Raises:
                None.

            Examples:
                None.
        """
            # Stack all of np arrays into one dimensional array (flattened)
        return np.vstack([self.word_averaging(glove_model, post) for post in text_list])
