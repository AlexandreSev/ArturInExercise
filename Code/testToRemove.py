# coding: utf-8

from utils import openPickle

sequences = openPickle("./Resources/trysequences.pkl")

for i, sequence in enumerate(sequences):
	print(sequence)
	if i > 50:
		break
