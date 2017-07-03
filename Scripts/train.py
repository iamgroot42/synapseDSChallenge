import models
import numpy as np
import sys
from keras.optimizers import SGD

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


def trainCombinedModel(x_train, y_train, x_val, y_val, x_test, y_test):
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model = models.combineModels(models.getSiftNetwork(), models.getDenseNetwork(), sgd)
	model.fit([x_train[:, :128], x_train[:, 128:]], y_train, epochs=20, batch_size=16, validation_data=([x_val[:, :128],x_val[:, 128:]], y_val))
	score = model.evaluate([x_test[:, :128],x_test[:, 128:]], y_test, batch_size=128)
	return score


if __name__ == "__main__":
	D, L = np.load(sys.argv[1]), np.load(sys.argv[2])
	X = np.concatenate((D, L), axis=1)
	Y = np.load(sys.argv[3])
	(a,b), (c,d), (e,f) = getSplitData(X, Y)
	print trainCombinedModel(a, b, c, d, e, f)	
