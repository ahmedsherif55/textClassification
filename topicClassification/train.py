from preprocessing import PreProcessing
from sklearn.model_selection import train_test_split
from Models import LR, SVM, NaiveBayes, Word2VecDeep

pre = PreProcessing()
data = pre.clean_text()
X = data['post']
y = data['tags']
# Split to 20% test data and 80% training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayes(X_train, X_test, y_train, y_test, pre.tags)
nb.train()

lr = LR(X_train, X_test, y_train, y_test, pre.tags)
lr.train()

#svm = SVM(X_train, X_test, y_train, y_test, pre.tags)
#svm.train()

wv = Word2VecDeep(X_train, X_test, y_train, y_test, pre.tags)
wv.train()

#bow = BOWDeep(X_train, X_test, y_train, y_test, pre.tags)
#bow.train()
