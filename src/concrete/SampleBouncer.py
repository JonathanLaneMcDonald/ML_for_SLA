
from interfaces.AbstractRestorableObject import AbstractRestorableObject

from numpy.random import random


class SampleBouncer(AbstractRestorableObject):
	def __init__(self, prepped_token_register = dict(), prepped_total_samples = 0):
		self.token_register = prepped_token_register
		self.total_samples_taken = prepped_total_samples

	def register_sampled(self, token):
		if token not in self.token_register:
			self.token_register[token] = 0
		self.token_register[token] += 1
		self.total_samples_taken += 1

	def accept_token(self, token):
		if self.total_samples_taken and token in self.token_register:
			global_average = self.total_samples_taken / len(self.token_register)
			return random() < (global_average / self.token_register[token])
		return True

	def get_token_register_size(self):
		return len(self.token_register)

	def get_state(self):
		return {
			'total_samples_taken': self.total_samples_taken,
			'token_register': {int(key): int(value) for key, value in self.token_register.items()}
		}

	@staticmethod
	def restore_state(state: dict):
		try:
			total_samples = state['total_samples_taken']
			token_register = {int(key): int(value) for key, value in state['token_register'].items()}
			return SampleBouncer(token_register, total_samples)
		except Exception as e:
			raise Exception("SampleBouncer::restore_state(): ", e)
