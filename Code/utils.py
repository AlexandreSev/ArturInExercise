# coding: utf-8

import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from preprocessing import toBoolList

def openPickle(path):
    """
    Return the object of the file.

    Parameters:
    -----------
        path (string): path to the pickle

    Returns:
    --------
        (?) What was inside the file
    """
    fr = open(path, "br")
    tmp = pickle.load(fr)
    fr.close()
    return tmp


def savePickle(path, obj):
    """
    Save the object into a binary pickle.

    Parameters:
    -----------
        path (str): where to save the object
        obj: object to save
    """
    # Check if the output directory exists
    tmp = path.split('/')
    outputDir = "/".join(tmp[:-1])
    
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    fw = open(path, "bw")
    pickle.dump(obj, fw)
    fw.close()


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


if __name__ == "__main__":
    if False:
        print(openPickle("./Data/dict.pkl"))

    if False:
        indices = getTrainTest("./Data/Learn/labels.pkl")
        labels = openPickle("./Data/Learn/labels.pkl")
        indices2 = getTrainTest(labels)
        indices3 = getTrainTest(toBoolList(labels))

        if np.all(indices[0] == indices2[0]) and np.all(indices[1] == indices2[1]):
            print("indices is equal to indices2")

        if np.all(indices[0] == indices3[0]) and np.all(indices[1] == indices3[1]):
            print("indices is equal to indices3")

        y = toBoolList(labels)
        if abs(np.mean(y[indices[0]]) - np.mean(y[indices[1]])) < 0.001:
            print("Stratify works")

    if True:
        obj = "Hello world!"
        path = "./IDoNotExist/reallyNot.pkl"
        savePickle(path, obj)
        print(openPickle(path))
        os.remove(path)
        os.rmdir("./IDoNotExist")        
        
        path = "./IExist/really.pkl"
        os.mkdir("./IExist")
        savePickle(path, obj)
        print(openPickle(path))
        os.remove(path)
        
        path = "./IExist/Almost/obj.pkl"
        savePickle(path, obj)
        print(openPickle(path))
        os.remove(path)
        os.rmdir("./IExist/Almost")
        os.rmdir("./IExist")
        