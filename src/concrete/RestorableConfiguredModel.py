
from keras.models import Model
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
		pass

	def restore_state(self, state: dict):
		pass



