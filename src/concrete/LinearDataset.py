
import numpy as np
from numpy.random import random

class LinearDataset:

	@staticmethod
	def find_next_delimiter(dataset, delimiter, position):
		while position < len(dataset) and dataset[position] != delimiter:
			position += 1
		return position

	def __init__(self, numpy_array_paths, entry_delimiter, validation_split = 0.10):
		self.dataset = np.array([], dtype=np.uint16)

		try:
			numpy_arrays = [np.frombuffer(open(path, 'rb').read(), dtype=np.uint16) for path in numpy_array_paths]
			self.dataset = np.concatenate(numpy_arrays)
			print(self.dataset.shape,'tokens loaded from',len(numpy_array_paths),'files')
		except Exception as e:
			print('Error Loading Dataset:', e)
			raise e

		self.validation_start = LinearDataset.find_next_delimiter(self.dataset, entry_delimiter, int(len(self.dataset)*(1-validation_split)))

	def __getitem__(self, key):
		return self.dataset[key]

	def random_training_position(self):
		return int(random()*self.validation_start)

	def random_validation_position(self):
		if len(self.dataset) - self.validation_start <= 0:
			raise Exception('Validation Data Requested, But Validation Dataset Size <= 0')
		return self.validation_start + int(random()*(len(self.dataset) - self.validation_start))

	def get_dataset_size(self):
		return len(self.dataset)
