# coding: utf-8

import numpy as np
import os
import word2vec

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D
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
                            input_length=X_train.shape[1])
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(512, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=["acc"])
model.summary()

modelSavePath = "./Resources/CNNWeight"
if not os.path.isdir(modelSavePath):
    os.mkdir(modelSavePath)

weightsFile = "512N_E_{epoch:02d}-VL_{val_loss:.2f}.h5"

callback = ModelCheckpoint(os.path.join(modelSavePath, weightsFile))

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256,
          callbacks=[callback])
