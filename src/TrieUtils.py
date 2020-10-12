
from SequenceTagger import SequenceTagger

import unidic

from fugashi import Tagger as FugashiTagger


def length_test(edict):
	"""Verify that the provided dictionary was properly tokenized by Fugashi

	1) We're expecting to see a distribution of sequence lengths (instead of all sequences having length==1), and
	2) Sequences longer than a certain length are dumped in the console so we can do a sanity check
	"""

	lengthTest = dict()
	for word in edict:
		if len(word) not in lengthTest:
			lengthTest[len(word)] = 0
		lengthTest[len(word)] += 1
		if len(word) >= 4:
			print(word)
	print(sorted(lengthTest.items()))


def get_spans(tagged_spans, numbers_suck=True):
	"""Receives an array of tuples from SequenceTagger and returns an array of contiguous spans"""

	spans = []
	span_start = 0
	for i in range(len(tagged_spans)):
		if tagged_spans[i][1]:
			if not i or (i and tagged_spans[i][1] == tagged_spans[i-1][1]):
				pass
			else:
				spans.append(''.join([x[0] for x in tagged_spans[span_start:i]]))
				span_start = i
		else:
			spans.append(''.join([x[0] for x in tagged_spans[span_start:i]]))
			span_start = i
	spans.append(''.join([x[0] for x in tagged_spans[span_start:]]))

	if numbers_suck:
		return ['#' if x.isdigit() else x for x in spans if len(x)]
	else:
		return [x for x in spans if len(x)]


def basic_trie_setup(edict, run_length_test=False):
	"""Receives a dictionary and initializes the taggers"""

	# initialize the fugashi wrapper with unidic
	fugashiMcTagger = FugashiTagger('-Owakati -d{}'.format(unidic.DICDIR))

	# parse unique edict keys from the edict file and then tokenize with fugashi/unidic
	edict_tokens = [fugashiMcTagger.parse(y).split() for y in edict]

	if run_length_test:
		length_test(edict_tokens)
	
	# initialize my Trie-based sequence tagger with the tokenized edict
	sequenceMcTagger = SequenceTagger(edict_tokens)

	return fugashiMcTagger, sequenceMcTagger
