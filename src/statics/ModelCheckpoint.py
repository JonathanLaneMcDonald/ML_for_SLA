
class ModelCheckpoint:

	"""Purpose of a ModelCheckpoint:
		To be able to access a model from an instant in time and either use it or resume training without any apparent interruption

		This implies that the model and peripherals are all stored in a way that facilitates functionally flawless reconstruction"""

	@staticmethod
	def save_checkpoint(model_checkpoint_base_path, model, data_generator, training_schedule, evaluation_metrics=[]):
		"""Going to need to implement get_state() type functions for all the things I want to be able to restore

		I may do this in the form of an interface with a function that needs to produce a dictionary of settings

		Then I'll probably just manually list all the things that need to be in the checkpoint to make a big dictionary and write a json file
		"""
		pass

	@staticmethod
	def load_checkpoint(checkpoint_path):
		"""Same as above, but with a restore_state() function"""

		pass

