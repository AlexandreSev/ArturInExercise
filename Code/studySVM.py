# coding: utf-8

import numpy as np
from os.path import join as pjoin

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils import openPickle, savePickle
from postprocessing import top20Coefs, convertLabels
from processing import evaluateModel, getPredictions
from preprocessing import TfIdfTransformer, sparseBagOfWords

encoder = openPickle("./Data/dict.pkl")
decoder = {encoder[i]: i for i in encoder}

# Naive Bayes with bag of words
#model = MultinomialNB(alpha=0.1)
model = LinearSVC(C = 0.1, class_weight="balanced")
_, expectedScore = evaluateModel(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl")
"""
print("")
print("Prob 0: %s. Prob 1: %s"%(np.exp(model.class_log_prior_[0]), np.exp(model.class_log_prior_[1])))
print("There are %s features."%len(model.feature_log_prob_[0]))
print("There are %s negatives sequences, %s positive"%(model.class_count_[0], model.class_count_[1]))

order1, order0 = top20Coefs(model)

print("\nPostive words with counts")
for ind in order1:
	print("%s: %s | %s"%(
		decoder[ind],
		model.feature_count_[0][ind],
		model.feature_count_[1][ind]))

print("")
print("Negative words with counts")
for ind in order0:
	print("%s: %s | %s"%(
		decoder[ind],
		model.feature_count_[0][ind],
		model.feature_count_[1][ind]))
"""
preds = getPredictions(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl",
					   "./Data/Test/sequences.pkl")

mcPreds = convertLabels(preds)

name = "model__%s__preprocesser__%s__expected__%.4f.pkl"%("LinearSVC(C=0.1, weight_class=balanced)",
														  "sparseBagOfWords",
														   expectedScore)

savePickle(pjoin("./Results/", name), mcPreds)

# Naive Bayes with tf-idf
print("\n" + "#"*50 + "\n")
#model = MultinomialNB(alpha=1.)
model = LinearSVC(C=0.001, class_weight="balanced")
preprocesser = TfIdfTransformer(norm=None)
_, expectedScore = evaluateModel(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl", preprocesser)
"""
print("")
print("Prob 0: %s. Prob 1: %s"%(np.exp(model.class_log_prior_[0]), np.exp(model.class_log_prior_[1])))
print("There are %s features."%len(model.feature_log_prob_[0]))
print("There are %s negatives sequences, %s positive"%(model.class_count_[0], model.class_count_[1]))

order1, order0 = top20Coefs(model)

print("\nPostive words with counts")
for ind in order1:
	print("%s: %s | %s"%(
		decoder[ind],
		model.feature_count_[0][ind],
		model.feature_count_[1][ind]))

print("")
print("Negative words with counts")
for ind in order0:
	print("%s: %s | %s"%(
		decoder[ind],
		model.feature_count_[0][ind],
		model.feature_count_[1][ind]))
"""
preds = getPredictions(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl",
					   "./Data/Test/sequences.pkl")

mcPreds = convertLabels(preds)

name = "model__%s__preprocesser__%s__expected__%.4f.pkl"%("LinearSVC(C=0.001, weight_class=balanced)",
														  preprocesser,
														  expectedScore)

savePickle(pjoin("./Results/", name), mcPreds)
