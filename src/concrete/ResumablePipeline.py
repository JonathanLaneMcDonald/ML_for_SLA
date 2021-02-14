
import os
import json


class ResumablePipeline:
	def __init__(self, status_path):
		self.status_path = status_path

		self.status = dict()
		if os.path.exists(self.status_path):
			try:
				with open(self.status_path, 'r') as json_file:
					self.status = json.load(json_file)
			except Exception as e:
				print('Error Reading Status from JSON:', e)

	def run_skippable(self, skippable_task, task_label):
		if (task_label not in self.status) or (self.status[task_label] != 'complete'):
			try:
				skippable_task()
				self.write_completed_task(task_label)
			except Exception as e:
				print('Error Completing Task:', e)
		else:
			print('Task Previously Completed (Skipping):', task_label)

	def write_completed_task(self, task_label):
		try:
			self.status[task_label] = 'complete'
			with open(self.status_path, 'w') as json_file:
				json.dump(self.status, json_file)
		except Exception as e:
			print('Error Writing Status to JSON', e)


