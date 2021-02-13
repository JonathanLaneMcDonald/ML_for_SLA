
class DictionaryBuilder:

	@staticmethod
	def sort_tokens_by_count(tokens):
		bycounts = list(reversed(sorted([(counts, key) for key, counts in tokens.items()])))
		return '\n'.join([str(x[0]) + '\t' + str(x[1]) for x in bycounts])

	@staticmethod
	def count_tokens_in_newline_delimited_path(tokenizer, plaintext_path, target_path):
		lines = 0
		vocab = dict()
		lengths = dict()
		with open(plaintext_path, 'r', encoding='utf-8', errors='ignore') as f:
			for line in f:

				line_tokens = tokenizer.tokenize(line[:-1])
				for token in line_tokens:
					if token not in vocab:
						vocab[token] = 0
					vocab[token] += 1

				doc_length = len(line_tokens)//1024
				if doc_length not in lengths:
					lengths[doc_length] = 0
				lengths[doc_length] += 1

				lines += 1
				if lines % 10000 == 0:
					print('(DictionaryBuilder::CountingTokens) Lines Processed:',lines,'Unique Tokens:',len(vocab),'Doc Lengths:',sorted(lengths.items()))
					open(target_path,'w',encoding='utf-8').write(DictionaryBuilder.sort_tokens_by_count(vocab))

		open(target_path,'w',encoding='utf-8').write(DictionaryBuilder.sort_tokens_by_count(vocab))
