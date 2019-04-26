from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.tags = tags
        self.model = None

    def train(self):
        # TfidfVectorizer applies CountVectorizer to count word frequencies then
        # applies TfidfTransformer to extract tfidf information for each token
        tfidf_vector = TfidfVectorizer()
        self.X_train = tfidf_vector.fit_transform(self.X_train)
        self.X_test = tfidf_vector.transform(self.X_test)

        """scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_micro': 'f1_macro'}


            scores = cross_validate(self.model,
                        self.X_train,
                        self.y_train,
                        cv=5,
                        scoring=scoring,
                        return_train_score=True)

        for metric_name in scores.keys():
            average_score = np.average(scores[metric_name])
            print('%s : %f' % (metric_name, average_score))"""

        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print('accuracy %s' % accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred, target_names=self.tags))


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
        self.model = SGDClassifier()
