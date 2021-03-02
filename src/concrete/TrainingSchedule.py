
from interfaces.AbstractRestorableObject import AbstractRestorableObject

class TrainingSchedule(AbstractRestorableObject):

	def __init__(self, current_loop=1, training_loops=1000, batch_size=128, training_batches=8192, validation_batches=128):
		self.config = {
			'current_loop': current_loop,
			'training_loops': training_loops,
			'batch_size': batch_size,
			'training_batches': training_batches,
			'training_samples': batch_size * training_batches,
			'validation_batches': validation_batches,
			'validation_samples': batch_size * validation_batches
		}

	def __getitem__(self, item):
		try:
			return self.config[item]
		except Exception as e:
			print('TrainingSchedule::__getitem__():', e)

	def increment_loop_counter(self):
		self.config['current_loop'] += 1

	def still_training(self):
		return self.config['current_loop'] < self.config['training_loops']




