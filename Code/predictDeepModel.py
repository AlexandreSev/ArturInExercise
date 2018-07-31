# coding: utf-8

from keras.models import load_model
from sklearn.metrics import f1_score

from utils import openPickle, savePickle
from preprocessing import getTrainTest, toBoolList
from preprocessing import sequencesCorrecter, preprocessDeepModel
from postprocessing import convertLabels

# Train Data

paddedTrainSeq = preprocessDeepModel("./Data/Learn/correctedSequences.pkl",
                                	 "./Data/Learn/kerasSequences.pkl",
                                	 409)
labels = toBoolList(openPickle("./Data/Learn/labels.pkl"))

trainInd, testInd = getTrainTest(labels)

X_train, X_val = paddedTrainSeq[trainInd], paddedTrainSeq[testInd]
y_train, y_val = labels[trainInd], labels[testInd]

# Test Data
sequences = openPickle("./Data/Test/sequences.pkl")
correcter = openPickle("./Resources/tokenCorrecter.pkl")
correctedSequences = sequencesCorrecter(sequences, correcter)

savePickle("./Data/Test/correctedSequences.pkl", correctedSequences)

paddedSeq = preprocessDeepModel("./Data/Test/correctedSequences.pkl",
                                "./Data/Test/kerasSequences.pkl",
                                409)

# CNN
CNNPath = "./Resources/CNNWeight/CNNWeight.h5"

cnn = load_model(CNNPath)

## Evaluate score

trainPreds = cnn.predict(X_train).flatten()
print("Training score: %.4f"%f1_score(trainPreds>0.5, y_train))

valPreds = cnn.predict(X_val).flatten()
print("Validation score: %.4f"%f1_score(valPreds>0.5, y_val))

preds = cnn.predict(paddedSeq).flatten()

## Predict the test set
preds = convertLabels((preds > 0.5).astype(int))
savePickle("./Results/cnnPreds.pkl", preds)

# LSTM
LSTMPath = "./Resources/LSTMWeight/LSTMWeight.h5"

lstm = load_model(LSTMPath)

## Evaluate score

trainPreds = lstm.predict(X_train).flatten()
print("Training score: %.4f"%f1_score(trainPreds>0.5, y_train))

valPreds = lstm.predict(X_val).flatten()
print("Validation score: %.4f"%f1_score(valPreds>0.5, y_val))

## Predict the test set
preds = lstm.predict(paddedSeq).reshape((-1, 1))

preds = convertLabels((preds > 0.5).astype(int))
savePickle("./Results/lstmPreds.pkl", preds)

