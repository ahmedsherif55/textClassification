from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from preprocessing import PreProcessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

class Model:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.tags = tags
        self.model = None

    def train(self, tfidf=True):
        # TfidfVectorizer applies CountVectorizer to count word frequencies then
        # applies TfidfTransformer to extract tfidf information for each token
        if tfidf:
            tfidf_vector = TfidfVectorizer()
            self.X_train = tfidf_vector.fit_transform(self.X_train)
            self.X_test = tfidf_vector.transform(self.X_test)

        scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro',
                   'f1_micro': 'f1_macro'}


        scores = cross_validate(self.model,
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring=scoring,
                    return_train_score=True)

        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print('accuracy %s' % accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred, target_names=self.tags))
        return scores


class NaiveBayes(Model):
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        Model.__init__(self, X_train, X_test, y_train, y_test, tags)
        self.model = MultinomialNB()


class LR(Model):
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        Model.__init__(self, X_train, X_test, y_train, y_test, tags)
        self.model = LogisticRegression(n_jobs=1, C=1e5, solver='lbfgs', multi_class='auto')


class SVM(Model):
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        Model.__init__(self, X_train, X_test, y_train, y_test, tags)
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)


class Word2VecDeep:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.y_train = y_train
        self.y_test = y_test
        self.tags = tags
        # Limit is used to get the most-frequent 500,000 word's vectors, so speed loading vectors a little.
        glove_model = KeyedVectors.load_word2vec_format("pretrained_vectors/gensim_glove_vectors.txt", binary=False,
                                                        limit=500000)
        # Used for initialization of model.syn0norm
        glove_model.init_sims(replace=True)

        print("Done Processing Pretrained Vectors")

        pre = PreProcessing()

        test_tokenized = X_test.apply(lambda item: self.tokenize_text(item))
        train_tokenized = X_train.apply(lambda item: self.tokenize_text(item))

        self.X_train_word_average = pre.word_averaging_list(glove_model, train_tokenized)
        self.X_test_word_average = pre.word_averaging_list(glove_model, test_tokenized)

        print("Done Applying Pretrained Vectors")

    def train(self):
        lr = LR(self.X_train_word_average, self.X_test_word_average, self.y_train, self.y_test, self.tags)
        return lr.train(tfidf=False)

    # Tokenize Word
    def tokenize_text(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                # To make sure that this is at least a word not single character
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens


class BOWDeep:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_words = 1000
        self.batch_size = 64
        self.epochs = 3
        self.num_classes = len(tags)

        # Build the model (Neural Network)
        # Sequential structure to store multiple sequential layers
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(self.max_words, )))
        self.model.add(Activation('relu'))
        # To prevent over fitting
        # Dropout drops 50% of information so not to over fit model and improve performance
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))
        # Categorical cross entropy loss because we have multiple classes
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        tokenize = text.Tokenizer(num_words=self.max_words, char_level=False)
        # Only fit on train, then use for testing
        tokenize.fit_on_texts(X_train)

        self.X_train = tokenize.texts_to_matrix(self.X_train)
        self.X_test = tokenize.texts_to_matrix(self.X_test)

        # label encoder to get one hot encoding for classes
        encoder = LabelEncoder()
        encoder.fit(self.y_train)
        self.y_train = encoder.transform(self.y_train)
        self.y_test = encoder.transform(self.y_test)

        self.y_train = utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = utils.to_categorical(self.y_test, self.num_classes)

        self.metrics = Metrics()

    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=True, validation_split=0.1,
                       callbacks=[self.metrics,
                                  EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        score = self.model.evaluate(self.X_test, self.y_test,
                                    batch_size=self.batch_size, verbose=True)
        print('Test accuracy:', score[1])
        return self.metrics.get_f1_scores()


class RNN:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = 5
        self.batch_size = 64
        self.num_classes = len(tags)
        # The maximum number of words to be used. Most frequeunt words
        MAX_NB_WORDS = 50000
        EMBEDDING_DIM = 100

        self.metrics = Metrics()

        self.model = Sequential()
        self.model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=self.X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(20, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=True, validation_split=0.1,
                       callbacks=[self.metrics,
                                  EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        score = self.model.evaluate(self.X_test, self.y_test,
                                    batch_size=self.batch_size, verbose=True)
        print('Test accuracy:', score[1])
        return self.metrics.get_f1_scores()


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: {:f} — val_precision: {:f} — val_recall {:f}".format(_val_f1, _val_precision, _val_recall))

    def get_f1_scores(self):
        return self.val_f1s
