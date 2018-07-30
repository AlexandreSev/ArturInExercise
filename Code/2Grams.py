# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from utils import openPickle
from preprocessing import toBoolList, getTrainTest

cv = CountVectorizer(ngram_range=(1, 2))

X = cv.fit_transform(openPickle("./Data/Learn/sentences.pkl"))

labels = toBoolList(openPickle("./Data/Learn/labels.pkl"))

trainInd, testInd = getTrainTest(labels)

X_train, X_test, y_train, y_test = X[trainInd], X[testInd], labels[trainInd], labels[testInd]

model = MultinomialNB(alpha=0.01)

model.fit(X_train, y_train)

trainScore = f1_score(y_train, model.predict(X_train))
testScore = f1_score(y_test, model.predict(X_test))

print("Training f1 score: %.4f"%trainScore)
print("Testing f1 score: %.4f"%testScore)