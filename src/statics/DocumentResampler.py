
class DocumentResampler:

	@staticmethod
	def resample(tokenizer, documents_path, resampled_documents_path, encodable_tokens_path):
		encodable_tokens = {key for key in open(encodable_tokens_path, 'r', encoding='utf-8').read().split('\n') if len(key)}

		total = 0
		saved = 0
		resampled_comments = open(resampled_documents_path, 'w', encoding='utf-8')
		with open(documents_path, 'r', encoding='utf-8', errors='ignore') as f:
			for line in f:
				total += 1
				if encodable_tokens.issuperset({tok for tok in tokenizer.tokenize(line[:-1])}):
					resampled_comments.write(line)

					saved += 1
					if saved % 10000 == 0:
						print('(DocumentResampler::Resample) Total Seen:', total, 'Total Saved:', saved)

		resampled_comments.close()

