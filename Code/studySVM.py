# coding: utf-8

import numpy as np
from os.path import join as pjoin

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils import openPickle, savePickle
from postprocessing import top20Coefs, convertLabels
from processing import evaluateModel, getPredictions
from preprocessing import TfIdfTransformer, sparseBagOfWords

# Naive Bayes with bag of words
model = LinearSVC(C = 0.1, class_weight="balanced")
_, expectedScore = evaluateModel(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl")

preds = getPredictions(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl",
                       "./Data/Test/sequences.pkl")

mcPreds = convertLabels(preds)

name = "model__%s__preprocesser__%s__expected__%.4f.pkl"%("LinearSVC(C=0.1, weight_class=balanced)",
                                                          "sparseBagOfWords",
                                                           expectedScore)

savePickle(pjoin("./Results/", name), mcPreds)

# Naive Bayes with tf-idf
print("\n" + "#"*50 + "\n")
model = LinearSVC(C=11., class_weight="balanced")
preprocesser = TfIdfTransformer(norm="l1")
_, expectedScore = evaluateModel(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl",
                                 preprocesser)

preds = getPredictions(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl",
                       "./Data/Test/sequences.pkl")

mcPreds = convertLabels(preds)

name = "model__%s__preprocesser__%s__expected__%.4f.pkl"
name = name%("LinearSVC(C=0.001, weight_class=balanced)", preprocesser, expectedScore)

savePickle(pjoin("./Results/", name), mcPreds)
