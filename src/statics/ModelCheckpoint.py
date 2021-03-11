
import json

from concrete.RestorableConfiguredModel import RestorableConfiguredModel
from concrete.ContextualEmbeddingsPreTrainingDataGenerator import ContextualEmbeddingsPreTrainingDataGenerator
from concrete.TrainingSchedule import TrainingSchedule


class ModelCheckpoint:

	"""Purpose of a ModelCheckpoint:
		To be able to access a model from an instant in time and either use it or resume training without any apparent interruption

		This implies that the model and peripherals are all stored in a way that facilitates functionally flawless reconstruction"""

	@staticmethod
	def save_checkpoint(model_checkpoint_base_path, restorable_model, data_generator, training_schedule, evaluation_metrics=[]):
		"""Going to need to implement get_state() type functions for all the things I want to be able to restore

		I may do this in the form of an interface with a function that needs to produce a dictionary of settings

		Then I'll probably just manually list all the things that need to be in the checkpoint to make a big dictionary and write a json file
		"""

		numbered_checkpoint_path = model_checkpoint_base_path + "checkpoint " + str(training_schedule.get_current_loop())
		latest_checkpoint_path = model_checkpoint_base_path + "latest checkpoint"

		checkpoint_config = {
			'restorable_model': restorable_model.get_state(),
			'data_generator': data_generator.get_state(),
			'training_schedule': training_schedule.get_state(),
			#'evaluation_metrics': [metric.get_state() for metric in evaluation_metrics]
		}

		try:
			with open(numbered_checkpoint_path, 'w') as json_file:
				json.dump(checkpoint_config, json_file)
			with open(latest_checkpoint_path, 'w') as json_file:
				json.dump(checkpoint_config, json_file)
		except Exception as e:
			raise Exception('Error Writing Model Checkpoints to JSON', e)

	@staticmethod
	def load_checkpoint(checkpoint_path):
		"""Same as above, but with a restore_state() function"""

		try:
			with open(checkpoint_path, 'r') as json_file:
				checkpoint = json.load(json_file)
			# these are currently hard-coded, too, but i'll need to make switches or something
			restorable_model = RestorableConfiguredModel.restore_state(checkpoint['restorable_model'])
			data_generator = ContextualEmbeddingsPreTrainingDataGenerator.restore_state(checkpoint['data_generator'])
			training_schedule = TrainingSchedule.restore_state(checkpoint['training_schedule'])
			return restorable_model, data_generator, training_schedule, []

		except Exception as e:
			raise Exception('Error Recreating Model Checkpoint from JSON:', e)

