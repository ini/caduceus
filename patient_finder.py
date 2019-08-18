import keras
import numpy as np
import os
import tensorflow as tf

from data import format_data
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerBlock
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)



### Constants

SAVED_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved')
INPUT_DIM = 65
SEQ_LENGTH = 168
MODEL_PATH = os.path.join(SAVED_DIR, 'mark16-032.h5')
ADMIT_EMBEDDING, DISCHARGE_EMBEDDING = np.load(os.path.join(SAVED_DIR, 'special_sequence_items.npy'))



### PatientFinder

class PatientFinder:

	def __init__(self, n_pca_components=32, n_knn_neighbors=10):
		self.pca = PCA(n_components=n_pca_components)
		self.knn = NearestNeighbors(n_neighbors=n_knn_neighbors, algorithm='auto')
		self.model = self.load_model(MODEL_PATH)


	def load_model(self, pretrained_model_path):
		pretrained_model = keras.models.load_model(pretrained_model_path, custom_objects={'discrete_logistic_mixture_NLL': lambda y_true, y_pred: 0 * y_pred})
		base_model = keras.models.Model(
			inputs=pretrained_model.get_layer('input_layer').input, 
			outputs=pretrained_model.get_layer('transformer4_normalization2').output)

		return base_model


	def fit_transform(self, data, patient_ids=None):
		data = format_data(data)
		check_data(data)

		X = pad(data)
		X = self.model.predict(X)
		X = X.reshape(-1, X.shape[-1])

		if patient_ids is None:
			patient_ids = np.arange(len(data))
			
		self.patient_ids = {i: patient_ids[patientint(i / SEQ_LENGTH)] for i in range(len(X))}
		self.patient_offsets = {i: patient_ids[patientint(i % SEQ_LENGTH)] for i in range(len(X))}

		self._fit_X = X
		self._fit_pca_X = self.pca.fit_transform(X)
		self.knn.fit(self._fit_pca_X)


	def transform(self, data)
		data = format_data(data)
		check_data(data)
		assert self.pca is not None and self.knn is not None, 'Must call function "fit_transform" before using "transform"'

		ends = np.clip([len(array) for array in data], 0, SEQ_LENGTH - 1)
		X = pad(data)
		X = self.model.predict(X)
		X = X[np.arange(len(X)), ends]

		return self.pca.transform(X) 


	def find_similar(self, data, num_similar=5, return_distance=False):
		data = format_data(data)
		check_data(data)

		X = self.transform(data)
		dist, ind = self.knn.kneighbors(X=X, n_neighbors=num_similar, return_distance=True) 

		if return_distance:
			return ind, dist

		return ind




DATA_PATH = '../data_utils/train_cls_data.npy'

def load_data(data_path):
	X = np.load(data_path)

	# Assertions
	assert len(X.shape) == 3, 'Expected array to have 3 dimensions, got {}'.format(len(X.shape))
	assert X.shape[-1] == INPUT_DIM, 'Expected input dimension to be {}, got {}'.format(INPUT_DIM, X.shape[-1])

	# Pad data
	pad_width = ((0, 0), (0, max(0, SEQ_LENGTH - X.shape[1])), (0, 0))
	X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

	return X[:, :SEQ_LENGTH]



X = load_data(DATA_PATH)
print(X.shape)

X = X[:, 1:, :]
print(X.shape)

p = PatientFinder()
p.fit_transform(X)

z = [np.ones((3, INPUT_DIM)), np.ones((12, INPUT_DIM))]
z[0][-1, :] = 99
z[1][3, :] = 98

print(p.find_similar(z))





