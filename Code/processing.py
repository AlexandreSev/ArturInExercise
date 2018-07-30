# coding: utf-8

import numpy as np

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

from utils import openPickle, getTrainTest
from preprocessing import toBoolList, sparseBagOfWords, TfIdfTransformer

def evaluateModel(model, trainPath, labelsPath, preprocesser=sparseBagOfWords):
	"""
	Evaluate the f1-score of a model.

	Parameters:	
	-----------
		model: Class with fit and predict methods.
		trainPath (str): The path of the pickle of the training examples.
		labelsPath (str): The path of the pickle of the training labels.
		preprocesser (func): Function used to transform the list of sequences into a matrix.
	"""
	sequences = np.array(openPickle(trainPath))
	labels = toBoolList(openPickle(labelsPath))
	return evaluateModel_(model, sequences, labels, preprocesser=preprocesser)
	

def evaluateModel_(model, sequences, labels, preprocesser=sparseBagOfWords):
	"""
	Evaluate the f1-score of a model.

	Parameters:	
	-----------
		model: Class with fit and predict methods.
		sequences (list<list<int>>): list of sequences. A sequence is a list of int.
		labels (list): Labels of the sequences.
		preprocesser (func): Function used to transform the list of sequences into a matrix.
	"""

	trainInd, testInd = getTrainTest(labels)
	trainSeq, trainLabels = preprocesser(sequences[trainInd]), labels[trainInd]

	n_features = trainSeq.shape[1]
	
	# If we use a bag of word, we have to give the shape of the matrix
	if isinstance(preprocesser, TfIdfTransformer) or 'shape' in preprocesser.__code__.co_varnames:
		testSeq = preprocesser(sequences[testInd], shape=(len(testInd), n_features))
	else:
		testSeq = preprocesser(sequences[testInd])
	
	testLabels = labels[testInd]

	print("Training...")
	model.fit(trainSeq, trainLabels)

	trainScore = f1_score(trainLabels, model.predict(trainSeq))
	testScore = f1_score(testLabels, model.predict(testSeq))

	print("Training f1 score: %.4f"%trainScore)
	print("Testing f1 score: %.4f"%testScore)
	return trainScore, testScore


def getPredictions(model, trainPath, labelsPath, testPath, preprocesser=sparseBagOfWords):
	"""
	Train a model and predict a testSet.

	Parameters:
	-----------
		model: class with fit and predict methods
		trainPath (str): The path of the pickle of the training examples.
		labelsPath (str): The path of the pickle of the training labels.
		testPath (str): The path of the pickle of the testing examples.
		preprocesser (func): Function used to transform  the list of sequences into a matrix.
	"""
	sequences = np.array(openPickle(trainPath))
	labels = toBoolList(openPickle(labelsPath))

	trainSeq = preprocesser(sequences)
	n_features = trainSeq.shape[1]

	model.fit(trainSeq, labels)

	testSeq = openPickle(testPath)
	return model.predict(preprocesser(testSeq, shape=(len(testSeq), n_features)))


if __name__ == "__main__":
	
	from utils import top20Coefs
	import numpy as np

	model = MultinomialNB()
	evaluateModel(model, "./Data/Learn/sequences.pkl", "./Data/Learn/labels.pkl")

	encoder = openPickle("./Data/dict.pkl")
	decoder = {encoder[i]: i for i in encoder}
	coefs = top20Coefs(model)

	print("Most positives:")
	for coef in coefs[0]:
		print("%s: %s"%(decoder[coef], np.exp(-model.feature_log_prob_[0][coef])))
	
	print("\nMost negatives:")
	for coef in coefs[1]:
		print("%s: %s"%(decoder[coef], np.exp(-model.feature_log_prob_[1][coef])))
