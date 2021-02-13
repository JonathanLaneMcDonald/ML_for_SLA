
import MeCab

from interfaces.AbstractTokenizer import AbstractTokenizer

class jpTokenizer(AbstractTokenizer):
	def __init__(self):
		self.wataki = MeCab.Tagger('-Owakati')

	def tokenize(self, s):
		return self.wataki.parse(s).split()
