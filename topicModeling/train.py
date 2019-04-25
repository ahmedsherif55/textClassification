from preprocessing import PreProcessing
from Dictionary import Dictionary

pre = PreProcessing()

data = pre.pre_process()

dict = Dictionary(data)

bow_corpus = dict.bag_of_words()

dict.print_sample_word_frequency(bow_corpus[0], 'bag')

corpus_tfidf = dict.tf_idf()

dict.print_sample_word_frequency(corpus_tfidf, 'tfidf')
