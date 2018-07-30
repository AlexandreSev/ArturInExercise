# coding: utf-8

from keras.models import load_model

from utils import openPickle, savePickle
from preprocessing import sequencesCorrecter, preprocessDeepModel
from postprocessing import convertLabels

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

preds = cnn.predict(paddedSeq).reshape((-1, 1))

preds = convertLabels((preds > 0.5).astype(int))
savePickle("./Results/cnnPreds.pkl", preds)

# LSTM
LSTMPath = "./Resources/LSTMWeight/LSTMWeight.h5"

lstm = load_model(LSTMPath)

preds = lstm.predict(paddedSeq).reshape((-1, 1))

preds = convertLabels((preds > 0.5).astype(int))
savePickle("./Results/lstmPreds.pkl", preds)

