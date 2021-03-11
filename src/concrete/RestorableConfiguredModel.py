
import time

from keras.models import Model, save_model, load_model
from interfaces.AbstractRestorableObject import AbstractRestorableObject

class RestorableConfiguredModel(AbstractRestorableObject):

	def __init__(self, model: Model, model_config: dict):
		self.model = model
		self.config = model_config

	def fit(self, **kwargs):
		return self.model.fit(**kwargs)

	def predict(self, **kwargs):
		return self.model.predict(**kwargs)

	def get_state(self):
		try:
			saved_model_path = "model checkpoint time="+str(time.time())
			save_model(saved_model_path)
			return {
				'saved_model_path': saved_model_path,
				'model_config': self.config
			}
		except Exception as e:
			print("RestorableConfiguratedModel::get_state(): ", e)

	@staticmethod
	def restore_state(state: dict):
		try:
			restorable_model = load_model(state['saved_model_path'])
			model_config = state['model_config']
			return RestorableConfiguredModel(restorable_model, model_config)
		except Exception as e:
			print("RestorableConfiguredModel::restore_state(): ", e)
