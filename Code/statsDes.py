# coding: utf-8

import pickle
import numpy as np
from sentenceFunctions import wordCount

from utils import openPickle

#Let's open the files
d = openPickle("./Data/dict.pkl")

print("We have %s different words."%len(d.keys()))
for i, key in enumerate(d.keys()):
	print("%s: %s"%(key, d[key]))
	if i > 2:
		break

d_reverse = {d[i]: i for i in d}

sentences = openPickle("./Data/Learn/sentences.pkl")

print("")
print("We have %s sentences"%len(sentences))
for i, sentence in enumerate(sentences):
	print(sentence)
	if i>2:
		break

sequences = openPickle("./Data/Learn/sequences.pkl")

print("")
print("We have %s sequences"%len(sequences))
for i, sequence in enumerate(sequences):
	print(sequence)
	if i>2:
		break

labels = openPickle("./Data/Learn/labels.pkl")

print("")
print("We have %s labels."%len(labels))
print(labels[0])
print(labels[45000])

#Now, let's compute some descriptive statistiques
print("")
print("We have %.4f %% of C in the database."%(np.mean(np.array(labels) == 'C') * 100))
print("We have %.4f %% of M in the database."%(np.mean(np.array(labels) == 'M') * 100))

print("\nWe have %s different words"%len(d.keys()), end=", ")

nWords = 0
for sequence in sequences:
	nWords += len(sequence)

print("and %s words in sequences."%nWords)

wc = wordCount(sequences)

countUnique = 0
for key in wc.keys():
	if wc[key] == 1:
		countUnique += 1

print("%s words appears only one time."%countUnique)

mask = np.array(labels) == 'M'
sequencesAr = np.array(sequences)
print("Average length of Mitterand's senteces: %s"%np.mean([len(s) for s in sequencesAr[mask]]))
print("Average length of Chirac's senteces: %s"%np.mean([len(s) for s in sequencesAr[np.logical_not(mask)]]))