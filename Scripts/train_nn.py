import models
from sklearn.utils import class_weight
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from keras.optimizers import SGD, Adam


def antiOneHot(y):
	return np.argmax(y, axis=1)


def getSplitData(ex, ey, splitRatio = 0.8):
	p = np.random.permutation(len(ex))
	X, Y = ex[p], ey[p]
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
	Dv, Lv, Yv = np.load(xdv), np.load(xsv), np.load(yv)
	Cv, C = np.concatenate((Dv, Lv), axis=1), np.concatenate((D, L), axis=1)
 	return getSplitData(np.concatenate((C, Cv)), np.concatenate((Y, Yv)), 0.7)


def trainCombinedModel(x_train, y_train, x_val, y_val, x_test, y_test, lr, bs, e):
	# opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model = models.combineModels(models.getSiftNetwork(), models.getDenseNetwork(), opt)
	classWeight = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train,1)), np.argmax(y_train,1))
	model.fit([x_train[:, :128], x_train[:, 128:]], y_train, epochs=e, batch_size=bs, validation_data=([x_val[:, :128],x_val[:, 128:]], y_val), class_weight=classWeight)
	model.save("dsml_model")
	score = model.evaluate([x_test[:, :128],x_test[:, 128:]], y_test, batch_size=128)
	predictions = model.predict([x_test[:, :128],x_test[:, 128:]])
	auc_score = roc_auc_score(y_test, predictions, average='weighted')
	return score, auc_score


if __name__ == "__main__":	
	# D, L = np.load("../Data/Xdeep.npy"), np.load("../Data/Xsift.npy")
	# X = np.concatenate((D, L), axis=1)
	# Y = np.load("../Data/Y.npy")
	# (a,b), (c,d), (e,f) = getSplitData(X, Y)
	(a,b), (c,d), (e,f) = loadFromFileUpdated("../Data/Xdeep.npy", "../Data/Xsift.npy", "../Data/Y.npy", "../Data/Xdeep_val.npy", "../Data/Xsift_val.npy", "../Data/Y_val.npy")
	scores = trainCombinedModel(a, b, c, d, e, f, float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
	print "\nTesting accuracy:", scores[0][1]
	print "AUC score:", scores[1]

