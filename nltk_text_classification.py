"""
Text classification is a machine learning technique that assigns a 
set of predefined categories to open-ended text. Text classifiers 
can be used to organize, structure, and categorize pretty much any 
kind of text from documents, medical studies and files, and all over the web.
"""


import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier # wrapper to include sklearn tool into nltk
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class Vote_Classifier(ClassifierI):
    
    def __init__(self, *classifiers) -> None:
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        return mode(vote)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        
        return conf


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
# collect all words from the movie review corpus 
for word in movie_reviews.words():
    all_words.append(word.lower())

# print(len(all_words))
# convert the all words into nltk frequence distribution
all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(60)) 
# print(all_words["stupid"])

# extract a lot of words that commonly used to train
word_features = list(all_words.keys())[:3000]

def find_features(document):
    """
    finds the common features from a given document 
    """
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    
    return features


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# convert the document into find_features and category
features = [(find_features(rev), category) for (rev, category) in documents]

training_set = features[:1900]
testing_set = features[1900:]

## Naive Bayes algorthim
# posterior = prior occurences * liklihood /evidence
# train classifier
# classifier = nltk.NaiveBayesClassifier.train(training_set)

# load the trained model
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# test classifier
print("Orginal Naive Bayes Algo accuracy:- {}".format((nltk.classify.accuracy(classifier, testing_set)*100)))
# classifier.show_most_informative_features(n=15)


# save the trained model
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
# print(len(documents))


# Multinomial Naive Bayes classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes classifier accuracy:- {}".format((nltk.classify.accuracy(MNB_classifier, testing_set)*100)))

# Gaussian Naive Bayes classifier
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("Gaussian Naive Bayes classifier accuracy:- {}".format((nltk.classify.accuracy(GNB_classifier, testing_set)*100)))

# Bernoulli Naive Bayes classifier
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernoulli Naive Bayes classifier accuracy:- {}".format((nltk.classify.accuracy(BNB_classifier, testing_set)*100)))

# Logistic Regression
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("Logistic Regression classifier accuracy:- {}".format((nltk.classify.accuracy(LR_classifier, testing_set)*100)))

# Stochastic Gradient Descent
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("Stochastic Gradient Descent classifier accuracy:- {}".format((nltk.classify.accuracy(SGD_classifier, testing_set)*100)))

# Linear Support Vector Classifier
LSVC_classifier = SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
print("Linear Support Vector classifier accuracy:- {}".format((nltk.classify.accuracy(LSVC_classifier, testing_set)*100)))

# C-Support Vector Classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("C-Support Vector classifier accuracy:- {}".format((nltk.classify.accuracy(SVC_classifier, testing_set)*100)))

# Nu-Support Vector Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("Nu-Support Vector classifier accuracy:- {}".format((nltk.classify.accuracy(NuSVC_classifier, testing_set)*100)))



voted_classifier = Vote_Classifier(classifier,
                                   MNB_classifier, 
                                   LR_classifier, 
                                   SGD_classifier, 
                                   LSVC_classifier, 
                                   SVC_classifier, 
                                   NuSVC_classifier)

print("********************************************************\n")
print("Voted classifier accuracy:- {}".format((nltk.classify.accuracy(voted_classifier, testing_set)*100)))



print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[1][0]), voted_classifier.confidence(testing_set[1][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[2][0]), voted_classifier.confidence(testing_set[2][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[3][0]), voted_classifier.confidence(testing_set[3][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[4][0]), voted_classifier.confidence(testing_set[4][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[5][0]), voted_classifier.confidence(testing_set[5][0])*100))
print("Classification: {} ** Confidence:- {}%".format(voted_classifier.classify(testing_set[6][0]), voted_classifier.confidence(testing_set[6][0])*100))


