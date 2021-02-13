
from numpy.random import random

class SampleBouncer:
	def __init__(self):
		self.token_register = dict()
		self.total_samples_taken = 0

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
