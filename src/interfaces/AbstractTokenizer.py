
class AbstractTokenizer:

	def tokenize(self, plaintext_document):
		'''Take a plaintext document and return a list of tokens'''
		raise Exception('AbstractTokenizer::tokenize not implemented')
