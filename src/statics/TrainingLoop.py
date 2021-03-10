
from statics.ModelCheckpoint import ModelCheckpoint


class TrainingLoop:

	@staticmethod
	def commence_training(model_checkpoint_base_path):
		latest_checkpoint_path = model_checkpoint_base_path + "latest checkpoint"
		model, data_generator, training_schedule, evaluation_metrics = ModelCheckpoint.load_checkpoint(latest_checkpoint_path)
		TrainingLoop.training_loop(model, data_generator, training_schedule, evaluation_metrics)

	@staticmethod
	def training_loop(restorable_model, data_generator, training_schedule, evaluation_metrics):
		while training_schedule.still_training():
			train_features, train_positions, train_labels = data_generator.generate(training_schedule['training_samples'])
			val_features, val_positions, val_labels = data_generator.generate(training_schedule['validation_samples'])
			restorable_model.fit(x=[train_features, train_positions], y=train_labels, batch_size=training_schedule['batch_size'], epochs=1,
				verbose=1, validation_data=([val_features, val_positions], val_labels))
			ModelCheckpoint.save_checkpoint(restorable_model, data_generator, training_schedule)
			training_schedule.increment_loop_counter()






