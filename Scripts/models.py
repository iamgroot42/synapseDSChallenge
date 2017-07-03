
from keras.layers import Input, concatenate, Dense
from keras.models import Model


def getDenseNetwork(act='relu'):
	main_input = Input(shape=(128,))
	x = Dense(16, activation=act)(main_input)
	x = Dense(4, activation=act)(x)
	return [main_input, x]


def getSiftNetwork(act='relu'):
	main_input = Input(shape=(128,))
	x = Dense(32, activation=act)(main_input)
	x = Dense(8, activation=act)(x)
	x = Dense(4, activation=act)(x)
	return [main_input, x]


def combineModels(siftModel, denseModel,opt, act='relu'):
	x = concatenate([siftModel[1], denseModel[1]])
	x = Dense(4, activation=act)(x)
	x = Dense(2, activation='softmax')(x)
	model = Model(inputs=[siftModel[0], denseModel[0]], output=x)
	print model.input_shape, model.output_shape
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

