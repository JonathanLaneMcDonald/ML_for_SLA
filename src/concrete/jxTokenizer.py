
import re
import nltk

from interfaces.AbstractTokenizer import AbstractTokenizer

class jxTokenizer(AbstractTokenizer):
	def __init__(self, deprotect=True, uncased=True, digits_to_hash=False, merge_hash_symbols=False):
		self.config = {
			'deprotect': deprotect,
			'uncased': uncased,
			'digits_to_hash': digits_to_hash,
			'merge_hash_symbols': merge_hash_symbols
			}

	def tokenize(self, s):
		if 'deprotect' in self.config and self.config['deprotect']:
			s = s.replace('[RET]','\n').replace('[TAB]','\t')
		if 'uncased' in self.config and self.config['uncased']:
			s = s.lower()
		if 'digits_to_hash' in self.config and self.config['digits_to_hash']:
			s = re.sub('[\d]','#',s)
			if 'merge_hash_symbols' in self.config and self.config['merge_hash_symbols']:
				s = re.sub('#{1,9}','#',s)
		return nltk.word_tokenize(s)
