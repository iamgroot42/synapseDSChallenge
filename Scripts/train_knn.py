import models
from sklearn.utils import class_weight
import numpy as np
import sys
from sklearn.metrics import roc_auc_score


def antiOneHot(y):
	return np.argmax(y, axis=1)


def getSplitData(X, Y, splitRatio = 0.8):
	indices = {}
	splitValRatio = splitRatio + (1.0 - splitRatio) / 2.0
	Y_labels = antiOneHot(Y)
	for classes in np.unique(Y_labels):
		indices[classes] = []
	for i in range(len(Y_labels)):
		indices[Y_labels[i]].append(i)
	splitIndices = [[], [], []]
	for label in indices.keys():
		splitIndices[0] += indices[label][:int(splitRatio * len(indices[label]))]
		splitIndices[1] += indices[label][int(splitRatio * len(indices[label])): int(splitValRatio * len(indices[label]))]
		splitIndices[2] += indices[label][int(splitValRatio * len(indices[label])):]
	X_train, Y_train = X[splitIndices[0]], Y[splitIndices[0]]
	X_val, Y_val = X[splitIndices[1]], Y[splitIndices[1]]
	X_test, Y_test = X[splitIndices[2]], Y[splitIndices[2]]
	return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def loadFromFile(x_base, y_base):
	X_train, Y_train = np.load(x_base + "train.npy"), np.load(y_base + "train.npy")
	X_val, Y_val = np.load(x_base + "val.npy"), np.load(y_base + "val.npy")
	X_test, Y_test = np.load(x_base + "test.npy"), np.load(y_base + "test.npy")
	return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def trainKNNModel(x_train, y_train, x_test, y_test, K):
	knn = models.getKNNModel(K)
	knn.fit(x_train, y_train)
	score = knn.score(x_test, y_test)
	predict = knn.predict(x_test)
	auc_score = roc_auc_score(y_test, predict, average='weighted')	
	return score, auc_score


if __name__ == "__main__":	
	D, L = np.load("../Data/Xdeep.npy"), np.load("../Data/Xsift.npy")
	X = np.concatenate((D, L), axis=1)
	Y = np.load("../Data/Y.npy")
	(a,b), (c,d), (e,f) = getSplitData(X, Y)
	A, B = np.concatenate((a,c)), np.concatenate((b,d))
	scores = trainKNNModel(A, B, e, f, int(sys.argv[1]))
	print "Testing accuracy:", scores[0]
	print "AUC score:", scores[1]
