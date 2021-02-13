
import numpy as np

from concrete.SequenceTagger import SequenceTagger

class DatasetEncoder:

	@staticmethod
	def auto_map_sorted_from_file(name):
		sorted_from_file = sorted({x for x in open(name, 'r', encoding='utf-8').read().split('\n') if len(x)})
		return {language:number for number, language in enumerate(sorted_from_file)}

	@staticmethod
	def prepare_bpe_tokens_from_file(bpe_token_path):
		bpe_tokens = DatasetEncoder.auto_map_sorted_from_file(bpe_token_path)
		for tok in ['[SEG]', '[PAD]', '[UNK]', '[MASK]']:
			bpe_tokens[tok] = len(bpe_tokens)
		return bpe_tokens

	@staticmethod
	def encode_dataset(tokenizer, resampled_documents_path, bpe_token_path, datablock_path, mappings_path, max_tokens_per_document, max_tokens_per_datablock):
		bpe_tokens = DatasetEncoder.prepare_bpe_tokens_from_file(bpe_token_path)

		sequenceMcTagger = SequenceTagger(bpe_tokens.keys())

		dynamic = dict()

		total_encoded = 0
		total_excluded = 0
		tokens_used = 0
		tokens = []
		block_number = 1
		with open(resampled_documents_path, 'r', encoding='utf-8') as f:
			for line in f:
				new_tokens = ['[SEG]']
				for tok in tokenizer.tokenize(line[:-1]):
					if tok not in dynamic:
						dynamic[tok] = sequenceMcTagger.bpeEncode(tok)
					new_tokens += dynamic[tok]
				new_tokens += ['[SEG]']

				if len(new_tokens) <= max_tokens_per_document:
					tokens += [bpe_tokens[x] for x in new_tokens]
					tokens_used += len(new_tokens)

					total_encoded += 1
					if total_encoded % 10000 == 0:
						print('(DatasetEncoder::EncodeDataset)', total_encoded, total_excluded, len(tokens), tokens_used)
				else:
					total_excluded += 1

				if len(tokens) > max_tokens_per_datablock:
					open(datablock_path+str(block_number), 'wb').write(np.array(tokens, dtype=np.uint16).tobytes())
					open(mappings_path, 'w', encoding='utf-8').write('\n'.join([key + '\t' + str(mapping) for key, mapping in bpe_tokens.items()]))
					block_number += 1
					tokens = []

		if len(tokens):
			open(datablock_path+str(block_number), 'wb').write(np.array(tokens, dtype=np.uint16).tobytes())
			open(mappings_path, 'w', encoding='utf-8').write('\n'.join([key + '\t' + str(mapping) for key, mapping in bpe_tokens.items()]))
