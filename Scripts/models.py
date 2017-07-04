
from keras.layers import Input, concatenate, Dense, Dropout
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def getDenseNetwork(act='relu'):
	main_input = Input(shape=(128,))
	x = Dense(256, activation=act)(main_input)
	x = Dropout(0.5)(x)
	x = Dense(64, activation=act)(x)
	x = Dropout(0.5)(x)
	x = Dense(16, activation=act)(x)
	return [main_input, x]


def getSiftNetwork(act='relu'):
	main_input = Input(shape=(128,))
	x = Dense(256, activation=act)(main_input)
	x = Dropout(0.5)(x)
	x = Dense(64, activation=act)(x)
	x = Dropout(0.5)(x)
	x = Dense(16, activation=act)(x)
	return [main_input, x]


def combineModels(siftModel, denseModel,opt, act='relu'):
	x = concatenate([siftModel[1], denseModel[1]])
	x = Dense(128, activation=act)(x)
	x = Dropout(0.5)(x)
	x = Dense(2, activation='softmax')(x)
	model = Model(inputs=[siftModel[0], denseModel[0]], output=x)
	print model.input_shape, model.output_shape
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def getKNNModel(k):
	neigh = KNeighborsClassifier(n_neighbors=k)
	return neigh 


def getSVMModel(kernel):
	clf = SVC(probability=True, kernel=kernel)
	return clf
