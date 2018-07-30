import numpy as np
import os
import word2vec

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from preprocessing import embeddingMatrix, reIndexToken, reIndexSequences, toBoolList
from utils import openPickle, savePickle, getTrainTest

np.random.seed(42)

modelPath = "./Resources/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

# Download the model if needed
if not os.path.isfile(modelPath):
    link = " http://embeddings.org/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
    os.system("wget -O " + modelPath + link)

# Load the model
w2v = word2vec.load(modelPath)
vocab = set(w2v.vocab)

# Load the encoder
encoder = openPickle("./Data/dict.pkl")
decoder = {encoder[key]: key for key in encoder}

sequencesPath = "./Data/Learn/kerasSequences.pkl"

#if not os.path.isfile(sequencesPath):
if True:   
    fromOldToNew = reIndexToken(w2v, decoder)

    newCoder = {"pad": 0, "unk": len(decoder) - 1}
    for key in decoder:
        if fromOldToNew[key] != len(decoder) - 1:
            newCoder[decoder[key]] = fromOldToNew[key]

    savePickle("./Data/newDict.pkl", newCoder)
    
    oldSeqPath = "./Data/Learn/correctedSequences.pkl"
    if not os.path.isfile(oldSeqPath):
        raise FileNotFoundError("Please run studyWord2Vec.py")
    
    sequences = openPickle(oldSeqPath)
    sequences = reIndexSequences(sequences, fromOldToNew)
    savePickle(sequencesPath, sequences)
else:
    sequences = openPickle(sequencesPath)

maxLength = max([len(seq) for seq in sequences])
paddedSeq = pad_sequences(sequences, maxlen=maxLength)

labels = np.array(toBoolList(openPickle("./Data/Learn/labels.pkl"))).astype(int)

print('Shape of data tensor:', paddedSeq.shape)
print('Shape of label tensor:', labels.shape)

trainInd, testInd = getTrainTest(labels)

X_train, X_val = paddedSeq[trainInd], paddedSeq[testInd]
y_train, y_val = labels[trainInd], labels[testInd]

encoder = openPickle("./Data/newDict.pkl")
decoder = {encoder[key]: key for key in encoder}

# Build the embedding matrix
embMatrix = embeddingMatrix(w2v, decoder)

embMatrix[-1, :] = np.mean(embMatrix[:-1], axis=0)

embedding_layer = Embedding(embMatrix.shape[0], embMatrix.shape[1], weights=[embMatrix],
                            input_length=maxLength)
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

callback = ModelCheckpoint(os.path.join(modelSavePath, "512N_trainable_E_{epoch:02d}-VL_{val_loss:.2f}.h5"))

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256,
          callbacks=[callback])
