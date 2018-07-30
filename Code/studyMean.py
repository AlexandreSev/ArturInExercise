# coding: utf-8

import numpy as np
import word2vec
import os

from xgboost.sklearn import XGBClassifier

from processing import evaluateModel
from preprocessing import getMeanVectors
from utils import openPickle, savePickle


if not os.path.isfile("./Data/Learn/embeddedMeanSequences.pkl"):
	encoder = openPickle("./Data/dict.pkl")
	decoder = {encoder[key]: key for key in encoder}

	w2v = word2vec.load("./Resources/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin")

	preprocesser = lambda x: getMeanVectors(x, w2v, decoder)

	sequences = np.array(openPickle("./Data/Learn/correctedSequences.pkl"))

	for i in range(len(sequences) // 5000):
		if i == 0:
			embeddedSeq = preprocesser(sequences[0:5000])
		else:
			embeddedSeq = np.vstack((embeddedSeq, preprocesser(sequences[5000 * i: 5000 * (i+1)])))
		print("Process until i = %s"%i)

	embeddedSeq = np.vstack((embeddedSeq, preprocesser(sequences[5000 * i:])))

	savePickle("./Data/Learn/embeddedMeanSequences.pkl", embeddedSeq)

model = XGBClassifier(n_estimators=500, max_depth=5, reg_alpha=1., reg_lambda=10.)
_, expectedScore = evaluateModel(model, "./Data/Learn/embeddedMeanSequences.pkl",
	"./Data/Learn/labels.pkl", lambda x:x)
print("")

