
from statics.DictionaryBuilder import DictionaryBuilder
from statics.CharacterBuilder import CharacterBuilder
from statics.TokenLearner import TokenLearner
from statics.DocumentResampler import DocumentResampler
from statics.DatasetEncoder import DatasetEncoder
from statics.ByteNetEncoder import ByteNetEncoder, ByteNetEncoderConfig
from concrete.ResumablePipeline import ResumablePipeline
from concrete.LinearDataset import LinearDataset
from concrete.ContextualEmbeddingsPreTrainingDataGenerator import ContextualEmbeddingsPreTrainingDataGenerator

class TrainingDataPrepPipeline(ResumablePipeline):
	"""This pipeline takes a tokenizer, a configuration, and a target corpus and returns a list of files ready for import and training"""

	@staticmethod
	def config_is_valid(config):

		required_keys = {	'min_chars_for_dict_entry',
							'min_tokens_for_dict_entry',
							'bpe_tokens_to_learn',
							'max_bpe_tokens_per_doc',
							'datablock_write_trigger_size',
							'model_input_size'		}

		for key in required_keys:
			if not config.__contains__(key) or not isinstance(config[key], int):
				return False

		return True

	def __init__(self, tokenizer, original_docs_path, config):

		if not TrainingDataPrepPipeline.config_is_valid(config):
			raise Exception('TrainingDataPrepPipeline::ConfigurationNotValid')

		status_tracker_path = original_docs_path + ' - status'
		super().__init__(status_tracker_path)

		self.execute(tokenizer, original_docs_path, config)

	def execute(self, tokenizer, original_docs_path, config):

		token_dictionary_path = original_docs_path + ' - token dictionary'
		character_dictionary_path = original_docs_path + ' - character dictionary'
		bpe_tokens_base_path = original_docs_path + ' - bpe tokens - '
		encodable_tokens_path = original_docs_path + ' - encodable tokens'
		resampled_document_path = original_docs_path + ' - resampled'
		training_dataset_base_path = original_docs_path + ' - dataset - '
		training_dataset_collection_path = original_docs_path + ' - dataset files'
		training_dataset_token_map_path = original_docs_path + ' - dataset bpe token mappings'

		self.run_skippable(
			lambda : DictionaryBuilder.count_tokens_in_newline_delimited_path(
				tokenizer,
				original_docs_path,
				token_dictionary_path),
			'DictionaryBuilder')

		self.run_skippable(
			lambda : CharacterBuilder.count_characters_in_counts_first_dictionary(
				token_dictionary_path,
				character_dictionary_path),
			'CharacterBuilder')

		self.run_skippable(
			lambda : TokenLearner.learn_tokens(
				(character_dictionary_path, config['min_chars_for_dict_entry']),
				(token_dictionary_path, config['min_tokens_for_dict_entry']),
				(bpe_tokens_base_path, config['bpe_tokens_to_learn']),
				encodable_tokens_path),
			'TokenLearner')

		self.run_skippable(
			lambda : DocumentResampler.resample(
				tokenizer,
				original_docs_path,
				resampled_document_path,
				encodable_tokens_path),
			'DocumentResampler')

		self.run_skippable(
			lambda : DatasetEncoder.encode_dataset(
				tokenizer,
				resampled_document_path,
				bpe_tokens_base_path + str(config['bpe_tokens_to_learn']),
				training_dataset_base_path,
				training_dataset_collection_path,
				training_dataset_token_map_path,
				config['max_bpe_tokens_per_doc'],
				config['datablock_write_trigger_size']),
			'DatasetEncoder')

		# set up some variables and then create a model checkpoint
