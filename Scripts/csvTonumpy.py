import csv
import sys
import numpy as np
from keras.utils import np_utils


def parseFile(filename,rowsThere=True):
	sift_data = []
	deep_data = []
	labels = []
	identifiers = []
	with open(filename, 'rb') as f:
		reader = csv.reader(f, delimiter=',')
		reader.next()
		for row in reader:
			identifiers.append(row[0])
			sift_data.append(row[1:129])
			deep_data.append(row[129:257])
			if rowsThere:
				labels.append(row[257])
	if not rowsThere:
		return identifiers, np.array(sift_data, dtype='int'), np.array(deep_data, dtype='float32')
	return identifiers, np.array(sift_data, dtype='int'), np.array(deep_data, dtype='float32'), np_utils.to_categorical(labels, 2).astype('int')


if __name__ == "__main__":
	filename = sys.argv[1]
	# I,S,D,L = parseFile(filename)
	I, S, D = parseFile(filename, False)
	np.save(sys.argv[2] + "Xdeep.npy", D)
	np.save(sys.argv[2] + "Xsift.npy", S)
	#np.save(sys.argv[2] + "Y.npy", L)

