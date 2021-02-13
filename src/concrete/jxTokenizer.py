
import re

from interfaces.AbstractTokenizer import AbstractTokenizer

class jxTokenizerConfig:
	'''
	jxTokenizerConfig exists so that I can have an easy way of versioning the jxTokenizer
	'''

	@staticmethod
	def load_from_file(configFilename):
		try:
			config = {}
			with open(configFilename, 'r', encoding='utf-8') as configFile:
				for entry in configFile:
					if len(entry) and entry[0] != '#':
						args = [x for x in entry.split() if len(x)]
						if len(args) == 3 and args[1] == '=':
							config[args[0]] = args[2]
			return config
		except Exception as e:
			print(e)

	@staticmethod
	def load_default_configuration():
		return {'deprotect':'true',
				'uncased':'true',
				'digits_to_hash':'false',
				'merge_hash_symbols':'false',
				'delimit_on_nonalpha':'true'}

class jxTokenizer(AbstractTokenizer):
	def __init__(self, tokenizerConfig = jxTokenizerConfig.load_default_configuration()):
		self.config = tokenizerConfig

	def tokenize(self, s):
		if 'deprotect' in self.config and self.config['deprotect'] == 'true':
			s = s.replace('[RET]','\n').replace('[TAB]','\t')
		if 'uncased' in self.config and self.config['uncased'] == 'true':
			s = s.lower()
		if 'digits_to_hash' in self.config and self.config['digits_to_hash'] == 'true':
			s = re.sub('[\d]','#',s)
			if 'merge_hash_symbols' in self.config and self.config['merge_hash_symbols'] == 'true':
				s = re.sub('#{1,9}','#',s)
		if 'delimit_on_nonalpha' in self.config and self.config['delimit_on_nonalpha'] == 'true':
			new = ''
			for c in s:
				if c.isalpha():
					new += c
				else:
					new += ' ' + c + ' '
			s = new
		return [x for x in s.split() if len(x)]
