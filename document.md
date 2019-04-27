# Algorithms Explanation

 
## SVM:
This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning, see the partial_fit method.
    For best results using the default learning rate schedule, the data should
    have zero mean and unit variance.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).
    
    In the linear classifier model, we assumed that training examples plotted in space. These data points are expected to be separated by an apparent gap. It predicts a straight hyperplane dividing 20 classes.
    The primary focus while drawing the hyperplane is on maximizing the distance from hyperplane to the nearest data point of either class. The drawn hyperplane called as a maximum-margin hyperplane
### The Accuracy is 81% .

## Logistic Regression:
is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).
Like all regression analyses, the logistic regression is a predictive analysis.
Logistic regression  is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    entropy loss if the 'multi_class' option is set to 'multinomial'.
    
### The Accuracy is 78%

## naive Bayes classifiers
are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem.

### The Accuracy is 74%
 
## Word2vec embeddings 
is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc
a method to construct such an embedding. It can be obtained using two methods (both involving Neural Networks): Skip Gram and Common Bag Of Words (CBOW)

1- Load pre trained "Glove" embeddings.
2- Tokenize words.
3- Retrieve normalized word vector for each word in each post and calculate mean of vectors (np arrays)
4- Stack all of np arrays into one dimensional array (flattened)
5- Fit data to Logistic Regression classifier and predict

### The Accuracy is 63%

## BOW with Keras
 
1- Separate the data into training and test sets.
2- Use tokenizer methods to count the unique words in our vocabulary and assign each of those words to indices.
3- Calling fit_on_texts() automatically creates a word index lookup of our vocabulary.
4- We limit our vocabulary to the top words by passing a num_words param to the tokenizer.
5- With our tokenizer, we can now use the texts_to_matrix method to create the training data that we’ll pass our model.
6- We feed a one-hot vector to our model.
7- After we transform our features and labels in a format Keras can read, we are ready to build our text classification model.
8- When we build our model, all we need to do is tell Keras the shape of our input data, output data, and the type of each layer. keras will look after the rest.
9- When training the model, we’ll call the fit() method, pass it our training data and labels, batch size and epochs.

### The Accuracy is 79%


## RNN
Recurrent Neural Networks (RNN) are a powerful and robust type of neural networks and belong to the most promising algorithms out there at the moment because they are the only ones with an internal memory.
LSTM’s enable RNN’s to remember their inputs over a long period of time. This is because LSTM’s contain their information in a memory, that is much like the memory of a computer because the LSTM can read, write and delete information from its memory.


## The Accuracy is 79%