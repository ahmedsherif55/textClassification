from preprocessing import PreProcessing
#from Models import NaiveBayes
from sklearn.model_selection import train_test_split
from Models import logisticR,SVM

data = PreProcessing()
#data.plot_figures()
data = data.clean_text()
X = data['post']
y = data['tags']
# Split to 20% test data and 80% training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#nb = NaiveBayes(X_train, X_test, y_train, y_test, data.tags)
#nb.train()

#lg = logisticR(X_train, X_test, y_train, y_test, data.tags)
#lg.train()

modle = SVM(X_train, X_test, y_train, y_test, data.tags)
modle.train()