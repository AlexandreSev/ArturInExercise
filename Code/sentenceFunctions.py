# coding: utf-8
from collections import defaultdict
import pickle

def wordCount(sequences):
	"""
	Create a dictionary with the number of occurence of each word
	
	Parameters:
	-----------
		sequences (list<list<int>>): List of sequences.

	Returns:
	--------
		(dict): mapping from word (e.g. an int) to the number of occurence
	"""

	d = defaultdict(int)
	for sequence in sequences:
		for word in sequence:
			d[word] += 1
	return d

def removeUniqueWords(sequences, wordOcc):
	"""
	Remove from the sequences the words that appears only one time.

	Parameters:
	-----------
		sequences (list<list<int>>): list of sequences. A sequence is a list of int.
		wordOcc (dict): mapping from words to number of occurences

	Returns:
	--------
		(list<list<int>>): list of sequences without words that appears only one time.
	"""

	answer = []
	for sequence in sequences:
		answer.append([word for word in sequence if wordOcc[word] > 1])
	return answer

if __name__ == "__main__":

	from utils import openPickle
	import numpy as np

	sequences = openPickle("./Data/Learn/sequences.pkl")

	wordOcc = wordCount(sequences)
	notUniqueSequences = removeUniqueWords(sequences, wordOcc)

	print(np.sum([len(sequence) == 0 for sequence in notUniqueSequences]))
