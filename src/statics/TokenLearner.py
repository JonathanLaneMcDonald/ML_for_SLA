
class WordCountsTuple:
	def __init__(self, counts, tokens):
		self.counts = counts
		self.tokens = tokens

	def __repr__(self):
		return '('+str(self.counts)+','+str(self.tokens)+')'

class TokenLearner:

	@staticmethod
	def pad_internals(string_as_list):
		return [string_as_list[x] if x == 0 else '##' + string_as_list[x] for x in range(len(string_as_list))]

	@staticmethod
	def register_token(dictionary, token, counts):
		if token not in dictionary:
			dictionary[token] = 0
		dictionary[token] += counts

	@staticmethod
	def extract_learned_tokens(list_of_wc):
		found_tokens = set()
		for wc in list_of_wc:
			for token in wc.tokens:
				found_tokens.add(token)
		return found_tokens

	@staticmethod
	def find_most_common_pair(list_of_wc):
		dictionary = dict()
		for wc in list_of_wc:
			for i in range(len(wc.tokens)-1):
				TokenLearner.register_token(dictionary, ''.join(wc.tokens[i:i+2]), wc.counts)
		bycounts = list(reversed(sorted([(counts, tup) for tup, counts in dictionary.items()])))
		return bycounts[0]

	@staticmethod
	def merge_pair(list_of_wc, mergeable):
		for wc in list_of_wc:
			updated = False
			for i in range(len(wc.tokens)-1):
				if ''.join(wc.tokens[i:i+2]) == mergeable:
					wc.tokens[i] = wc.tokens[i] + wc.tokens[i+1].replace('##','')
					wc.tokens[i+1] = '[DEL]'
					updated = True
			if updated:
				wc.tokens = [x for x in wc.tokens if x != '[DEL]']

	@staticmethod
	def extract_supported_words(list_of_wc, bpe_tokens):
		supported_words = set()
		for wc in list_of_wc:
			if bpe_tokens.issuperset(set(wc.tokens)):
				supported_words.add(''.join(wc.tokens))
		return supported_words

	@staticmethod
	def init_word_prep(x):
		return WordCountsTuple(int(x.split()[0]), list(TokenLearner.pad_internals(x.split()[1])))

	@staticmethod
	def load_token_elements(element_path, occurrence_cutoff):
		return {x.split()[1] for x in open(element_path, 'r', encoding='utf-8').read().split('\n') if len(x) and int(x.split()[0]) >= occurrence_cutoff}

	@staticmethod
	def load_tokens(token_path, elements, occurrence_cutoff):
		learnable_tokens = []
		encodable_tokens = []
		with open(token_path, 'r', encoding='utf-8') as f:
			for line in f:
				if len(line.split()) == 2:
					counts, word = line.split()

					if int(counts) >= occurrence_cutoff and elements.issuperset({x for x in word}):
						learnable_tokens.append(TokenLearner.init_word_prep(line))
						encodable_tokens.append(word)

		return learnable_tokens, encodable_tokens

	@staticmethod
	def commence_learning(learnable_tokens, bpe_target_path, bpe_token_count):
		learned_tokens = len(TokenLearner.extract_learned_tokens(learnable_tokens))
		while learned_tokens < (bpe_token_count+1):
			mergeable = TokenLearner.find_most_common_pair(learnable_tokens)
			TokenLearner.merge_pair(learnable_tokens, mergeable[1])

			learned_tokens += 1
			if learned_tokens % 1000 == 0 or learned_tokens == bpe_token_count:
				open(bpe_target_path+str(learned_tokens),'w',encoding='utf-8').write('\n'.join(sorted(TokenLearner.extract_learned_tokens(learnable_tokens))))

			print('(TokenLearner::CommenceLearning: Newest:',mergeable,'Learned Tokens:',learned_tokens,'Supported Words:',len(learnable_tokens))

	@staticmethod
	def learn_tokens(element_information, token_information, bpe_information, encodable_tokens_target_path):
		elements = TokenLearner.load_token_elements(element_information[0], element_information[1])
		print(len(elements),'token components loaded')

		learnable_tokens, encodable_tokens = TokenLearner.load_tokens(token_information[0], elements, token_information[1])
		print(len(encodable_tokens),'encodable tokens loaded')

		open(encodable_tokens_target_path, 'w', encoding='utf-8').write('\n'.join(encodable_tokens))

		TokenLearner.commence_learning(learnable_tokens, bpe_information[0], bpe_information[1])
