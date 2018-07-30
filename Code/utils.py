# coding: utf-8

import pickle
import numpy as np
import os

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

if __name__ == "__main__":
    if False:
        print(openPickle("./Data/dict.pkl"))

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
        