# Imports
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split


class PreProcessing:
    def __init__(self):
        data = pd.read_csv('dataset/abcnews-date-text.csv')
        # Remove unwanted column
        data.__delitem__('publish_date')
        # Test only with 100 samples TODO: Fix here and split data to train and test
        self.headlines = data['headline_text'][:100]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        self.filtered_words = []

    def pre_process(self):
        """  Filter text and tokenize sentences to remove some noise from the data.

            This function filter remove numbers, leading & trailing whitespaces,
            lower words and remove stop words, then tokenize sentences (split them to words).

            Args:
                None.

            Returns:
                list of lists: Each list contains filtered words that represent one data item.

            Raises:
                None.

            Examples:
                None.
        """
        self.remove_numbers()
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in self.headlines]
        lower_sentences = []
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        for sentence in tokenized_sentences:
            # convert words to lower case
            lower_sentences.append([lemmatizer.lemmatize(word.lower()) for word in sentence if word is not "" and word not in stop_words])

        return lower_sentences

    def remove_numbers(self):
        """  Filter text to remove numbers.

            This function make a transition table with punctuation symbols, then it uses a regular
            expression to substitute each number with null (empty string).

            Args:
                None.

            Returns:
                None. This function overwrite self.headlines list directly

            Raises:
                None.

            Examples:
                None.
        """
        new = []
        for sentence in self.headlines: # TODO: Convert to list comprehension
            translator = sentence.maketrans('', '', string.punctuation)
            sentence = sentence.translate(translator)
            sentence = re.sub(r'\d+', "", sentence)
            new.append(sentence)
        self.headlines = new
