# coding: utf-8

import numpy as np

def top20Coefs(model):
    """
    Return the 10 most positive and the 10 most negative words.

    If we note P(c|w) the probability of a class knowing a word, c1(c0) the positive (negative)
    class, the coefficient associate to a word for a class ci is

                        Xi = P(ci|w) / (P(c1|w) + P(c0|w)).
    
    If we apply a logarithm, we have 

                        Xi = log(P(ci|w)) - log(P(c0|w) + P(c1|w))

    Parameters:
    -----------
        model: class with the attribute coef_

    Returns:
    --------
        ((list<int>, list<int>)): a pair of list. The first one correpond to the most positive
            features, the second one to the most negative
    """

    X0 = model.feature_log_prob_[0] - np.log(np.sum(np.exp(model.feature_log_prob_), axis=0))
    X1 = model.feature_log_prob_[1] - np.log(np.sum(np.exp(model.feature_log_prob_), axis=0))
    
    order0 = np.argsort(X0)
    order1 = np.argsort(X1)

    return order1[-1:-11:-1], order0[-1:-11:-1]

def convertLabels(y):
	"""
	Transform 0/1 labels into the "C"/"M" labels

	Parameters:
	-----------
		y (np.array): binary labels

	Returns:
	--------
		(np.array): 'C'/'M' labels
	"""
	return ['M' if p == 1 else 'C' for p in y]

if __name__ == "__main__":
	y = [0, 1, 0, 1]
	print(convertLabels(y))
