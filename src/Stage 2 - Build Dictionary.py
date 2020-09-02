
from TrieUtils import get_spans, basic_trie_setup

# parse entries out of the utf-8 encoded version of edict and make a set out of the entries
edict = {x.split('/')[0].split('[')[0] for x in open('edict.utf8','r',encoding='utf-8').read().split('\n') if len(x)}

# provide that set of dictionary entries to set up the taggers
fugashiMcTagger, sequenceMcTagger = basic_trie_setup(edict)

lines = 0
vocab = dict()
with open('japanese comments','r',encoding='utf-8') as f:
	for line in f:

		# parse the current line (minus the \n at the end) with fugashi
		# then use sequenceMcTagger to label the sequence with edict
		# then get the array of spans so we can keep track of words:counts for resampling
		for span in get_spans(sequenceMcTagger.tagDocument(fugashiMcTagger.parse(line[:-1]).split())):
			if span not in vocab:
				vocab[span] = 0
			vocab[span] += 1

		# and periodically show a progress report because this is going to take a while :D
		lines += 1
		if lines % 100000 == 0:
			print (lines,len(vocab))

# then dump the sequences into a dictionary file, sorting by counts
bycounts = list(reversed(sorted([(counts, key) for key, counts in vocab.items()])))
open('japanese comments dictionary','w',encoding='utf-8').write('\n'.join([str(x[0]) + '\t' + str(x[1]) for x in bycounts]))
