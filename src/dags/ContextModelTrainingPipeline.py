
from statics.DictionaryBuilder import DictionaryBuilder
from statics.CharacterBuilder import CharacterBuilder
from statics.TokenLearner import TokenLearner
from statics.DocumentResampler import DocumentResampler
from statics.DatasetEncoder import DatasetEncoder
from statics.ByteNetEncoder import ByteNetEncoder, ByteNetEncoderConfig
from statics.ModelCheckpoint import ModelCheckpoint
from statics.TrainingLoop import TrainingLoop
from concrete.ResumablePipeline import ResumablePipeline
from concrete.LinearDataset import LinearDataset
from concrete.ContextualEmbeddingsPreTrainingDataGenerator import ContextualEmbeddingsPreTrainingDataGenerator
from concrete.TrainingSchedule import TrainingSchedule


class ContextModelTrainingPipeline(ResumablePipeline):
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

	@staticmethod
	def create_initial_model_checkpoint(training_dataset_collection_path, training_dataset_token_map_path, config, model_checkpoint_base_path):
		# initialize a LinearDataset object and then initialize a DataGenerator object
		token_map = {word: int(token) for word, token in
					 [x.split() for x in open(training_dataset_token_map_path, 'r', encoding='utf-8').read().split('\n') if len(x)]}

		restorable_model = ByteNetEncoder.get_model(ByteNetEncoderConfig.modified_test(len(token_map), config['model_input_size']))

		dataset_files = [x for x in open(training_dataset_collection_path,'r').read().split('\n') if len(x)]
		linear_dataset = LinearDataset(dataset_files, token_map['[SEG]'])
		data_generator = ContextualEmbeddingsPreTrainingDataGenerator(linear_dataset, token_map, config['model_input_size'])

		schedule = TrainingSchedule(training_batches=128, validation_batches=16)

		ModelCheckpoint.save_checkpoint(model_checkpoint_base_path, restorable_model, data_generator, schedule)

	def __init__(self, tokenizer, original_docs_path, config):

		if not ContextModelTrainingPipeline.config_is_valid(config):
			raise Exception('ContextModelTrainingPipeline::ConfigurationNotValid')

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
		model_checkpoint_base_path = original_docs_path + ' - model checkpoint - '

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

		self.run_skippable(
			lambda : ContextModelTrainingPipeline.create_initial_model_checkpoint(
				training_dataset_collection_path,
				training_dataset_token_map_path,
				config,
				model_checkpoint_base_path),
			'CreateInitialCheckpoint')

		self.run_skippable(
			lambda : TrainingLoop.commence_training(
				model_checkpoint_base_path),
			'TrainingLoop')
