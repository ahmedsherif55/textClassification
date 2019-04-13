import gensim
from gensim import models
from pprint import pprint


class Dictionary:
    def __init__(self, data):
        self.data = data
        self.dict = gensim.corpora.Dictionary(self.data)

    def bag_of_words(self):
        """  Run bag of words model on dataset.

            This function filter words that is repeated at least one and no more than 50% of training size,
            then it doc2bow function that produce gensim corpora (dict).

            Args:
                None.

            Returns:
                dictionary: Each list contains filtered words that represent one data item.

            Raises:
                None.

            Examples:
                None.
        """
        # Word should be repeated not less than 10% of training size and not more than 50%
        self.dict.filter_extremes(no_below=1, no_above=0.5, keep_n=100000)
        bow_corpus = [self.dict.doc2bow(doc) for doc in self.data]
        return bow_corpus

    def tf_idf(self):
        """  Run tf*idf model on dataset.

            This function uses gensim corpora (dict) to calculate term frequency (tf) which is
            frequency of words in the current document, then it calculates frequency of word in other
            documents.

            Args:
                None.

            Returns:
                list of tuples: Each element contain word id and frequency

            Raises:
                None.

            Examples:
                None.
        """
        bow_corpus = self.bag_of_words()
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        return corpus_tfidf

    def print_sample_word_frequency(self, corpus_item, type):
        if type == 'bag':
            for i in range(len(corpus_item)):
                print("Word {} (\"{}\") appears {} time.".format(corpus_item[i][0],
                                                                 self.dict[corpus_item[i][0]],
                                                                 corpus_item[i][1]))
        elif type == 'tfidf':
            for doc in corpus_item:
                pprint(doc)
                break
