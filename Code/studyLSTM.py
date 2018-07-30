# coding: utf-8

import numpy as np
import os
import word2vec

from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

from preprocessing import embeddingMatrix, preprocessDeepModel
from utils import openPickle

X_train, X_val, y_train, y_val = preprocessDeepModel("./Data/Learn/kerasSequences.pkl")
w2v = word2vec.load("./Resources/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin")

encoder = openPickle("./Data/newDict.pkl")
decoder = {encoder[key]: key for key in encoder}

# Build the embedding matrix
embMatrix = embeddingMatrix(w2v, decoder)

embMatrix[-1, :] = np.mean(embMatrix[:-1], axis=0)

embedding_layer = Embedding(embMatrix.shape[0], embMatrix.shape[1], weights=[embMatrix],
                            input_length=X_train.shape[1], trainable=True)

sequence_input = Input(shape=(X_train.shape[1],), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(128)(embedded_sequences)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

modelSavePath = "./Resources/LSTMWeight"
if not os.path.isdir(modelSavePath):
    os.mkdir(modelSavePath)

weightsFile = "128N_E_{epoch:02d}-VL_{val_loss:.2f}.h5"

callback = ModelCheckpoint(os.path.join(modelSavePath, weightsFile))

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=256,
          callbacks=[callback])
