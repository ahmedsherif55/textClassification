from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

import numpy as np


class SVM:
    def __init__(self, X_train, X_test, y_train, y_test, tags):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.tags = tags

        # Sequentially apply a list of transforms and a final estimator.
        # Intermediate steps of pipeline must implement fit and transform methods.
        # The final estimator only needs to implement fit.
        self.model = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                           ])

    def train(self):
        self.model.fit(self.X_train, self.y_train)

        #scoring = ['precision', 'recall', 'f1']
        scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}

        scores = cross_validate(self.model,
                                self.X_train,
                                self.y_train,
                                cv=10,
                                scoring=scoring,
                                return_train_score=True)

        #print('Train score is: %.5f' % scores.mean())
        for metric_name in scores.keys():
            average_score = np.average(scores[metric_name])
            print('%s : %f' % (metric_name, average_score))

        #nb.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        #print('accuracy %s' % precision_score(self.y_test, y_pred, average=None))
        print(classification_report(self.y_test, y_pred, target_names=self.tags))
