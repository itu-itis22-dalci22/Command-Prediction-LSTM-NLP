import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import random
from nltk.classify.scikitlearn import SklearnClassifier
import re
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode
import pickle
from TurkishStemmer import TurkishStemmer
from zemberek import (
    TurkishSentenceNormalizer,
    TurkishMorphology,
    TurkishTokenizer,
)

tokenizer = TurkishTokenizer.DEFAULT
turkish_stopwords = stopwords.words("turkish")
stemmer = TurkishStemmer()
morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("output.csv", "r", encoding="utf-8").read()

all_words = []

# Regex pattern to clean unwanted characters but keep spaces, letters, numbers, Turkish characters, and commas
pattern = r"\s+[^\sa-zA-Z0-9ğüşöçıĞÜŞİÖÇ,]"

# Cleaning the entire document content
document = re.sub(pattern, "", documents_f)

documents = []

# documents_mov = open("documents.pickle", "rb")
# documents = pickle.load(documents_mov)
# documents_mov.close()


# Splitting each line and ensuring we only split on the last comma
for r in document.split("\n"):
    if "," in r:
        # Splitting only on the last comma
        command_part, label = r.rsplit(",", 1)
        pattern_new = r"[^\sa-zA-Z0-9ğüşöçıĞÜŞİÖÇ]"
        command_part = re.sub(pattern_new, "", command_part)
        command_part = normalizer.normalize(command_part)
        command_part = command_part.lower()
        documents.append((command_part, label))
        tokens = tokenizer.tokenize(command_part)
        words = [
            stemmer.stem(token.content)
            for token in tokens
            if token.content not in turkish_stopwords
        ]
        all_words.extend(words)

random.shuffle(documents)

# documents_mov = open("documents.pickle", "wb")
# pickle.dump(documents, documents_mov)
# documents_mov.close()

# all_words_mov = open("all_words.pickle", "rb")
# all_words = pickle.load(all_words_mov)
# all_words_mov.close()

all_words = nltk.FreqDist(all_words)


# all_words_mov = open("all_words.pickle", "wb")
# pickle.dump(all_words, all_words_mov)
# all_words_mov.close()

word_features = []

# word_features_mov = open("word_features.pickle", "rb")
# word_features = pickle.load(word_features_mov)
# word_features_mov.close()

word_features = list(all_words.keys())[:400]

# word_features_mov = open("word_features.pickle", "wb")
# pickle.dump(word_features, word_features_mov)
# word_features_mov.close()

# featuresets = []

# featuresets_mov = open("featuresets.pickle", "rb")
# featuresets = pickle.load(featuresets_mov)
# featuresets_mov.close()


def find_features(command):
    command = normalizer.normalize(command)
    tokens = tokenizer.tokenize(command)
    words = [stemmer.stem(token.content) for token in tokens]
    features = {}
    for w in word_features:
        features[w] = w in words
    return features


# command_part = input("Enter a command: ")
# featuresets = [(find_features(command_part), "voleybol")]

# testing_set = featuresets

featuresets = [
    (find_features(command_part), label) for (command_part, label) in documents
]

command_with_label = [(command_part, label) for (command_part, label) in documents]

# featuresets_mov = open("featuresets.pickle", "wb")
# pickle.dump(featuresets, featuresets_mov)
# featuresets_mov.close()


training_set = featuresets[:400]
testing_set = featuresets[400:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.train(training_set)

# classifier_f = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, classifier_f)
# classifier_f.close()

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()
# print("------------------------------------------------------- \n \n")
# print(classifier.classify(find_features(command_part)))
# print(" \n \n -------------------------------------------------------")

testing_set_special = command_with_label[400:]

i = 0
for test_command in testing_set_special:
    testing_state = [testing_set[i]]
    accuracy = (nltk.classify.accuracy(classifier, testing_state)) * 100

    if accuracy < 50:
        print(
            f"command: {test_command[0]} predict label: {classifier.classify(find_features(test_command[0]))}  real label: {test_command[1]}"
        )
    # print(
    #     "Original Naive Bayes Algo accuracy percent:",
    #     (nltk.classify.accuracy(classifier, testing_state)) * 100,
    # )
    i += 1

print(
    "Original Naive Bayes Algo accuracy percent:",
    (nltk.classify.accuracy(classifier, testing_set)) * 100,
)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

# # MNB_classifier_f = open("MNB_classifier.pickle", "wb")
# # pickle.dump(MNB_classifier, MNB_classifier_f)
# # MNB_classifier_f.close()

# # MNB_classifier_f = open("MNB_classifier.pickle", "rb")
# # MNB_classifier = pickle.load(MNB_classifier_f)
# # MNB_classifier_f.close()
print(
    "MNB_classifier accuracy percent:",
    (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100,
)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
# # BernoulliNB_classifier_f = open("BernoulliNB_classifier.pickle", "wb")
# # pickle.dump(BernoulliNB_classifier, BernoulliNB_classifier_f)
# # BernoulliNB_classifier_f.close()

# # BernoulliNB_classifier_f = open("BernoulliNB_classifier.pickle", "rb")
# # BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
# # BernoulliNB_classifier_f.close()
print(
    "BernoulliNB_classifier accuracy percent:",
    (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100,
)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
# # LogisticRegression_classifier_f = open("LogisticRegression_classifier.pickle", "wb")
# # pickle.dump(LogisticRegression_classifier, LogisticRegression_classifier_f)
# # LogisticRegression_classifier_f.close()

# # LogisticRegression_classifier_f = open("LogisticRegression_classifier.pickle", "rb")
# # LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
# # LogisticRegression_classifier_f.close()
print(
    "LogisticRegression_classifier accuracy percent:",
    (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100,
)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
# # SGDClassifier_classifier_f = open("SGDClassifier_classifier.pickle", "wb")
# # pickle.dump(SGDClassifier_classifier, SGDClassifier_classifier_f)
# # SGDClassifier_classifier_f.close()

# # SGDClassifier_classifier_f = open("SGDClassifier_classifier.pickle", "rb")
# # SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
# # SGDClassifier_classifier_f.close()
print(
    "SGDClassifier_classifier accuracy percent:",
    (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100,
)

voted_classifier = VoteClassifier(
    classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier,
    SGDClassifier_classifier,
)

print(
    "voted_classifier accuracy percent:",
    (nltk.classify.accuracy(voted_classifier, testing_set)) * 100,
)
