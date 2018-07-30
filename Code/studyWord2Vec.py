# coding: utf-8

import word2vec
import os
import numpy as np
import unidecode

from utils import openPickle, savePickle
from preprocessing import sequencesCorrecter

modelPath = "./Resources/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

# Download the model if needed
if not os.path.isfile(modelPath):
    link = " http://embeddings.org/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
    os.system("wget -O " + modelPath + link)

# Load the model
model = word2vec.load(modelPath)
vocab = set(model.vocab)

# Load the encoder
encoder = openPickle("./Data/dict.pkl")
decoder = {encoder[key]: key for key in encoder}

# Get the words not present in the model
if not os.path.isfile("./Resources/missingWords.pkl"):
    count = 0
    missingWords = []
    for key in encoder:
        if key not in vocab:
            missingWords.append(key)
            count += 1

    print("There are %s words not present in the model."%count)

    # Save them
    savePickle("./Resources/missingWords.pkl", missingWords)
else:
    missingWords = openPickle("./Resources/missingWords.pkl")

# Study these missing words
nNumber = 0
nPlurial = 0

print(">>>> Counting words")
for word in missingWords:
    if word.isdigit():
        nNumber += 1
    elif word[-1] == 's':
        nPlurial += 1

print("There are %s numbers and %s plurials."%(nNumber, nPlurial))

# Open training sequences
sequences = openPickle("./Data/Learn/sequences.pkl")

# Count missingWords
missingInt = set([encoder[word] for word in missingWords])
count = 0
total = 0
for sequence in sequences:
    for word in sequence:
        if word in missingInt:
            count += 1
    total += len(sequence)

print("We have %s missing Words in the training set, and %s words in total."%(count, total))

print(">>>>>>>> Correction")
correcter = {}

print(">>>> Passing to singular")

tmp = []
for word in missingWords:
    if word[-1] == 's' and word[:-1] in vocab and word[:-1] in encoder:
        correcter[encoder[word]] = [encoder[word[:-1]]]
    else:
        tmp.append(word)
print("Passing to singular help %s words."%(len(missingWords) - len(tmp)))
missingWords = tmp

print(">>>> Leading and trailing '")

tmp = []
for word in missingWords:
    correct = word.strip("'")
    if correct in vocab and correct in encoder:
        correcter[encoder[word]] = [encoder[correct]]
    else:
        tmp.append(word)
print("We can fix %s words by removing leading and trailing '"%(len(missingWords) - len(tmp)))
missingWords = tmp

print(">>>> Splitting with '")

tmp = []
for word in missingWords:
    if "'" in word:
        splitted = word.split("'")
        if np.all([w in vocab and w in encoder for w in splitted]):
            correcter[encoder[word]] = [encoder[w] for w in splitted]
        else:
            tmp.append(word)
    else:
        tmp.append(word)
print("We can fix %s words by splitting on '"%(len(missingWords) - len(tmp)))
missingWords = tmp

print(">>>> Removing accent")

tmp = []
for word in missingWords:
    correct = unidecode.unidecode(word)
    if correct in vocab and correct in encoder:
        correcter[encoder[word]] = [encoder[correct]]
    else:
        tmp.append(word)
print("We can fix %s words by removing accent"%(len(missingWords) - len(tmp)))
missingWords = tmp

print(">>>> Removing ponctuation.")

tmp = []
toRemove = []
for word in missingWords:
    if len(word) == 1 and not word.isalpha() and not word.isdigit():
        correcter[encoder[word]] = []
        toRemove.append(encoder[word])
    else:
        tmp.append(word)
print("We can remove %s ponctuations"%(len(missingWords) - len(tmp)))
missingWords = tmp

print(">>>> Result")
missingInt = set([encoder[word] for word in missingWords])
count = 0
total = 0
for sequence in sequences:
    for word in sequence:
        if word in toRemove:
            total -= 1
        elif word in missingInt:
            count += 1
    total += len(sequence)

print("We have %s missing Words in the training set, and %s words in total."%(count, total))

savePickle("./Resources/tokenCorrecter.pkl", correcter)

print(">>>> Apply the correction ...")
correctedSentences = sequencesCorrecter(sequences, correcter)
savePickle("./Data/Learn/correctedSequences.pkl", correctedSentences)
