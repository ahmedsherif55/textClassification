import nltk
from sklearn.model_selection import train_test_split
from preprocessing import PreProcessing
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import KeyedVectors

# Limit is used to get the most-frequent 500,000 words' vectors, so speed loading vectors a little
glove_model = KeyedVectors.load_word2vec_format("pretrained_vectors/gensim_glove_vectors.txt", binary=False, limit=500000)
glove_model.init_sims(replace=True)
print("Done")

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


pre = PreProcessing()

data = pre.getRawData()

# Split to 20% test data and 80% training data
train, test = train_test_split(data, test_size=0.2, random_state=42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values

X_train_word_average = pre.word_averaging_list(glove_model, train_tokenized)
X_test_word_average = pre.word_averaging_list(glove_model, test_tokenized)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train.tags)
y_pred = logreg.predict(X_test_word_average)
print('accuracy %s' % accuracy_score(test.tags, y_pred))
print(classification_report(test.tags, y_pred, target_names=pre.tags))
