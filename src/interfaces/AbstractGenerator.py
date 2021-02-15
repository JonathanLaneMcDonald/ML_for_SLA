
class AbstractGenerator:

	def generate(self, num_samples_to_generate, is_for_validation=False):
		'''Produce and return the specified number of samples'''
		raise Exception('AbstractGenerator::generate not implemented')
