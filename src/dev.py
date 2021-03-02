
from concrete.jpTokenizer import jpTokenizer
from concrete.jxTokenizer import jxTokenizer

from dags.TrainingDataPrepPipeline import TrainingDataPrepPipeline


def get_test_config():
	return {
				'min_chars_for_dict_entry': 1,
				'min_tokens_for_dict_entry': 10,
				'bpe_tokens_to_learn': 5000,
				'max_bpe_tokens_per_doc': 2**14,
				'datablock_write_trigger_size': 2**20,
				'model_input_size': 128
			}


def get_default_config():
	return {
				'min_chars_for_dict_entry': 1,
				'min_tokens_for_dict_entry': 10,
				'bpe_tokens_to_learn': 30000,
				'max_bpe_tokens_per_doc': 2**14,
				'datablock_write_trigger_size': 2**30,
				'model_input_size': 128
			}


TrainingDataPrepPipeline(jxTokenizer(), 'reddit comments dev', get_test_config())



