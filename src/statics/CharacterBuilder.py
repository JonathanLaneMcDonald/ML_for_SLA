
class CharacterBuilder:

	@staticmethod
	def sort_tokens_by_count(tokens):
		bycounts = list(reversed(sorted([(counts, key) for key, counts in tokens.items()])))
		return '\n'.join([str(x[0]) + '\t' + str(x[1]) for x in bycounts])

	@staticmethod
	def count_characters_in_counts_first_dictionary(dictionary_path, target_path):
		characters = dict()
		with open(dictionary_path, 'r', encoding='utf-8') as f:
			for line in f:
				if len(line.split()) == 2:
					counts, word = line.split()

					for c in word:
						if c not in characters:
							characters[c] = 0
						characters[c] += int(counts)

		open(target_path,'w',encoding='utf-8').write(CharacterBuilder.sort_tokens_by_count(characters))
