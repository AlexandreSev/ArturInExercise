# coding: utf-8

import numpy as np
import word2vec
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from scipy import sparse

from sentenceFunctions import wordCount
from utils import openPickle, savePickle

def getTrainTest(labels):
    """
    Split the dataset into a training and a testing set, conserving the labels.

    Parameters:
    -----------
        labels (string or list): Path to the pickle of the labels, or directly the labels.

    Returns:
    --------
        ((list<int>, list<int>)): Return a pair of lists of indices. The first list correspond to
            the training set, the second one to the testing set.
    """
    if type(labels) == str:
        y = toBoolList(openPickle(labels))
    else:
        y = toBoolList(labels)

    X = np.arange(len(y))
    return train_test_split(X, stratify=y, random_state=42)


def sparseBagOfWords(sequences, shape=None):
    """
    Create a sparse matrix representin the bag of words of a list of sequences

    Parameters:
    -----------
        sequences (list<list<int>>): list of sequences. A sequence is a list of int.
        shape ((int, int)): shape of the matrix. If None, will be set to
                            (len(sequences), max(sequences) + 1)

    Returns:
    --------
        (scipy.sparse.csr_matrix): sparse bag of words
    """

    if sequences == []:
        raise ValueError("Empty sequences")

    indices = []
    data = []
    indptr = [0]
    maxInd = 0

    for i, sequence in enumerate(sequences):
        
        wc = wordCount([sequence])
        for word in wc:
            # Do not append the word if it is outside the range
            if shape is None or word < shape[1]:
                indices.append(word)
                data.append(wc[word])
        
        indptr.append(len(indices))

        maxInd = max(maxInd, max(sequence))
    
    if shape is None:
        shape = (len(sequences), maxInd+1)
    elif shape[1] < maxInd + 1:
        raise ValueError("column index exceeds matrix dimensions")

    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def toBoolList(labels):
    """
    Tranform a list of labels into a list of True, False.

    Parameters:
    -----------
        labels(list): list with only two unique values

    Returns:
    --------
        (list<bool>): list of positives/negatives. There can not be more negatives than positives.
    """

    modalities = np.unique(labels)
    
    if len(modalities) != 2:
        raise ValueError ("We found %s modalitiesin labels, 2 expected."%len(modalities))

    answer = np.array(labels) == modalities[0]

    if np.mean(answer) > 0.5:
        answer = np.logical_not(answer)

    return answer


class TfIdfTransformer:
    """
    Wrapper of a tf-idf

    Attributes:
    -----------
        initialised (bool): set to True after the first call.
        n_features (int): number of features found during the first call, or imposed by the user.
        transformer (sklearn TfidfTransformer): tf-idf transformer
        memory (dict): memory of the init arguments
    """
    def __init__(self, n_features=None, *args, **kwargs):
        """
        Parameters:
        -----------
            n_features (int): Number of features. If None, it will be set to the maximum of the
                              train sequences.
            *args, **kwargs: arguments for the sklearn tdidfTransformer.
        """
        self.initialised = False
        self.n_features = n_features
        self.transformer = TfidfTransformer(*args, **kwargs)
        self.memory = {"n_features": n_features}

    def __call__(self, sequences, shape=None):
        """
        The first time this class is called, it will fit the sequences and transform them. After, it
        will just tranform the sequences.

        Parameters:
        -----------
            sequences (list<list<int>>): list of sequences. A sequence is a list of int.
            shape ((int, int)): shape of the returned matrix.

        Returns:
        --------
            (scipy.sparse.csr_matrix): tf-idf representation of the sequences.
        """
        if shape is not None:
            if ((self.n_features is not None and self.n_features != shape[1]) or
                    (shape[0] != len(sequences))):
                raise ValueError("Wrong shape")

            self.n_features = shape[1]

        if self.initialised:
            bow = sparseBagOfWords(sequences, (len(sequences), self.n_features))
            return self.transformer.transform(bow)
        
        else:
            if self.n_features is None:
                bow = sparseBagOfWords(sequences)
            else:
                bow = sparseBagOfWords(sequences, (len(sequences), self.n_features))

            self.n_features = bow.shape[1]
            
            self.initialised = True
            
            return self.transformer.fit_transform(bow)

    def reset(self):
        """
        Reset to the initial state
        """
        self.initialised = False
        self.n_features = self.memory["n_features"]

    def __str__(self):
        """
        Create an explicit string
        """
        return "TfIdfTransformer(norm=%s)"%self.transformer.norm


def sequencesCorrecter(sequences, correcter):
    """
    Change the token number in the sequences according to the correcter.

    Parameters:
    -----------
        sequences (list<list<int>>): list of sequences. A sequence is a list of token number.
        correcter (dict): mapping from a word to correct to a list of replacement tokens.

    Returns:
    --------
        (list<list<int>>): List of corrected sequences
    """
    answer = []
    for sequence in sequences:
        tmp = []
        for word in sequence:
            tmp += correcter.get(word, [word])
        answer.append(tmp)
    return answer


def getMeanVector(sequence, w2v, decoder):
    """
    Get the mean of the vectors of the words in a sequence.

    Parameters:
    -----------
        sequence (list<int>): list of token number
        w2v (word2vec.model): model with the embedding of the words
        decoder (dict): mapping form token number to word.  

    Returns:
    --------
        (np.array): mean vector of the sequence 
    """
    answer = np.zeros(w2v[w2v.vocab[0]].shape)
    vocab = set(w2v.vocab)
    count = 0

    for word in sequence:
        if decoder[word] in vocab:
            answer += w2v[decoder[word]]
            count +=1

    return answer / count


def embeddingMatrix(w2v, decoder):
    """
    Get the matrix of embedding.

    Parameters:
    -----------
        w2v (word2vec.model): model with the embedding of the words
        decoder (dict): mapping form token number to word.

    Returns:
    --------
        (np.array): matrix of emmbeddings of size(nVocab, dimEmb)
    """
    embeddings = np.zeros((len(decoder.keys()), 200))
    vocab = set(w2v.vocab)
    for key in decoder.keys():
        if decoder[key] in vocab:
            embeddings[key, :] = w2v[decoder[key]]

    return embeddings


def getMeanVectors(sequences, w2v, decoder):
    """
    Get the list of the mean vectors of each sequence in sequences.

    Parameters:
    -----------
        sequences (list<list<int>>): list of sequence
        w2v (word2vec.model): model with the embedding of the words
        decoder (dict): mapping form token number to word.

    Returns:
    --------
        (list<np.array>): mean vectors of the sequences
    """
    embeddings = embeddingMatrix(w2v, decoder)

    embeddedSeq = np.zeros((len(sequences), len(decoder.keys())))
    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print("Proccess %s rows"%i)
        for j in seq:
            embeddedSeq[i, j] += 1
        if len(seq) != 0:
            embeddedSeq[i] /= len(seq)
    
    return np.dot(embeddedSeq, embeddings)


def reIndexToken(w2v, decoder):
    """
    Recreate the code to remove words that are not in the pretrained embedding.

    Paramaters:
    -----------
        w2v (word2vec.model): model with the embedding of the words
        decoder (dict): mapping form token number to word.

    Returns:
    --------
        (dict): mapping from the old code to the new one. 0 is reserved to padding token,
                the max number to the unknown token.
    """
    vocab = set(w2v.vocab)
    fromOldToNew = {}
    unknownTokens = []

    for key in decoder.keys():
        if decoder[key] in vocab:
            fromOldToNew[key] = len(fromOldToNew.keys()) + 1
        else:
            unknownTokens.append(key)

    unknownIndex = len(fromOldToNew.keys()) + 1
    for token in unknownTokens:
        fromOldToNew[token] = unknownIndex
    return fromOldToNew


def reIndexSequences(sequences, fromOldToNew):
    """
    Transform the sequences according to the decoder.

    Parameters:
    -----------
        sequences (list<list<int>>): list of sequence
        fromOldToNew (dict): mapping from the old code to the new one.

    Returns:
    --------
        (list<list<int>>): Reindexed sequences
    """
    return [[fromOldToNew[key] for key in seq] for seq in sequences]

def preprocessDeepModel(sequencesPath):
    """
    Preprocess the sequences to make them trainable by a deep model.

    Parameters:
    -----------
        sequencesPath (str); where are stored the sequences

    Returns:
    --------
        (np.arrays): the training and the validation data, and the training and the validation
                     labels.
    """

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

    #if not os.path.isfile(sequencesPath):
    if not os.path.isfile(sequencesPath):   
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

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":

    if False:
        tmp = sparseBagOfWords(sequences)
        print(tmp)
    
    if False:
        np.random.seed(42)
        labels = (np.random.randint(0, 2, 100) + np.random.randint(0, 2, 100)) // 2
        print(np.mean(labels))

        boolLabels = toBoolList(labels)
        if np.all(labels.astype(bool) == boolLabels):
            print("Test 1: OK")

        labels = ['M' if  i == 1 else 'C' for i in labels]
        boolLabels = toBoolList(labels)
        if np.all((np.array(labels) == 'M') == boolLabels):
            print("Test 2: OK")

    if True:
        sequences = [[0, 1, 2], [2, 0, 5], [1], [2, 3, 2]]
        tfidf = TfIdfTransformer()
        print(tfidf(sequences))
        
        print("")
        sequencesbis = [[2, 3], [4, 0], [1, 1]]
        print(tfidf(sequencesbis))
        
        #Expect an error
        print("")
        sequenceErr = [[6]]
        print(tfidf(sequenceErr))
        