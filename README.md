# Text Classification

is one of the important and typical task in supervised machine learning (ML). Assigning categories to documents, which can be a web page, library book, media articles, gallery etc. has many applications like e.g. spam filtering, email routing, sentiment analysis etc. In this article, I would like to demonstrate how we can do text classification using python, scikit-learn and little bit of NLTK.

#### Dataset: 
We are using a relatively large data set of Stack Overflow questions and tags. The data is available in Google BigQuery,
We have over 10 million words in the data, The classes are very well balanced.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### Numpy:
sudo pip install -U numpy
#### sklearn:
pip install sklearn
#### Models:
pip install pymodels
#### Matplotlib :
python -m pip install -U matplotlib
#### NLTK :
pip install -U nltk
#### beautifulsoup 'bs4': 
pip install beautifulsoup4
#### Pandas:
pip install pandas



## Deployment

### class NaiveBayes:
NaiveBayes:
these are supervised learning methods based on applying Bayes' theorem with strong (naive) feature independence assumptions.
 
#### tf*idf :
 Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.
    
#### CountVectorizer    
    Convert a collection of text documents to a matrix of token counts

### class PreProcessing:
#### clean_text:
HTML decoding using lxml

#### filter_data:
1- Remove parentheses, brackets and special symbols from input.
2- Doesn't remove numbers, lower case characters, #, +, _ or space;(because they are found in calsses such as: "c++, c#")    
3- Transform words to lower case letters, make lemmetization & Filter stop words from text

## Built With

Anaconda The open-source [Anaconda Distribution](https://www.anaconda.com/distribution/) the easiest way to perform Python/R data science and machine learning on Linux, Windows, and Mac OS X

## Algorithms Explanation
you can find them through [document.md](document.md).

## Authors

* **Ahmed Sherif** - *Initial work* - [Ahmed sherif](https://github.com/ahmedsherif55)
* **Ahmed Khaled** - *Initial work* - [Ahmed Khaled](https://github.com/AhmedKhaledAbdalla)
* **Ahmed Samir** - *Initial work* - [Ahmed Samir](https://github.com/AhmedSamir848)
* **Khaled Amin** - *Initial work* - [Khalid Amin]()
* **Menna Mohamed** - *Initial work* - [Menna Mohamed]()

See also the list of [contributors](https://github.com/ahmedsherif55/textClassification/graphs/contributors) who participated in this project.

## License

This project is licensed under the FCIH License - see the [LICENSE.md](LICENSE.md) file for details
