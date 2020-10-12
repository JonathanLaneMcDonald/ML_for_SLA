from TrieUtils import get_spans, basic_trie_setup

import numpy as np

# define how many words to use in the training dataset and how long sequences can be for resampling
resample = 65000
min_length, max_length = 8, 1024
dictionary = {key for _, key in [
	x.split() for x in open('japanese comments dictionary', 'r', encoding='utf-8').read().split('\n')[:resample]]}

# provide that set of dictionary entries to set up the taggers
fugashiMcTagger, sequenceMcTagger = basic_trie_setup(dictionary)

# assign numerical values to dictionary entries and prepare to count occurrences in the final dataset
mapping = {key: index for index, key in enumerate(sorted(dictionary))}
counts = {key: 0 for key in mapping.keys()}

total = 0
lines = 0
tokens = []
with open('japanese comments', 'r', encoding='utf-8') as f:
	for line in f:
		total += 1

		# redo all the work from Stage 2 because I'm lazy
		# apply acceptance criteria for this sequence
		# write the numerical values the model will see and increment the counter
		spans = get_spans(sequenceMcTagger.tagDocument(fugashiMcTagger.parse(line[:-1]).split()))
		if min_length <= len(spans) < max_length and dictionary.issuperset(spans):
			tokens += [mapping[x] for x in spans] + [0xffff]

			for span in spans:
				counts[span] += 1

			lines += 1
			if lines % 10000 == 0:
				print(total, lines, len(tokens))

# write the dataset and supporting info (mapping and counts)
open('japanese comments dataset', 'wb').write(np.array(tokens, dtype=np.uint16).tobytes())
open('japanese comments dataset supporting info', 'w', encoding='utf-8').write(
	'\n'.join([x + '\t' + str(mapping[x]) + '\t' + str(counts[x]) for x in mapping.keys()]))
