import models
from sklearn.utils import class_weight
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from keras.optimizers import SGD, Adam


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


def loadFromFileUpdated(xd, xs, y, xdv, xsv, yv):
	D, L, Y = np.load(xd), np.load(xs), np.load(y)
	Dv, Lv, Y_test = np.load(xdv), np.load(xsv), np.load(yv)
	X_test = np.concatenate((Dv, Lv), axis=1)
 	(X_train, Y_train), (p, q), (r, s) = getSplitData(np.concatenate((D,L),axis=1), Y)
	X_val = np.concatenate((p,r))
	Y_val = np.concatenate((q,s))
	return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def trainCombinedModel(x_train, y_train, x_val, y_val, x_test, y_test, kernel):
	svm = models.getSVMModel(kernel)
	svm.fit(np.concatenate((x_train, x_val)), antiOneHot(np.concatenate((y_train, y_val))))
	score = svm.score(x_test, antiOneHot(y_test))
	predictions = svm.predict(x_test)
	auc_score = roc_auc_score(y_test, predictions, average='weighted')
	return score, auc_score


if __name__ == "__main__":	
	# D, L = np.load("../Data/Xdeep.npy"), np.load("../Data/Xsift.npy")
	# X = np.concatenate((D, L), axis=1)
	# Y = np.load("../Data/Y.npy")
	# (a,b), (c,d), (e,f) = getSplitData(X, Y)
	(a,b), (c,d), (e,f) = loadFromFileUpdated("../Data/Xdeep.npy", "../Data/Xsift.npy", "../Data/Y.npy", "../Data/Xdeep_val.npy", "../Data/Xsift_val.npy", "../Data/Y_val.npy")
	scores = trainCombinedModel(a, b, c, d, e, f, sys.argv[1])
	print "\nTesting accuracy:", scores[0][1]
	print "AUC score:", scores[1]
