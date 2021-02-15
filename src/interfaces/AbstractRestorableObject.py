
class AbstractRestorableObject:

	def get_state(self):
		"""Produce a dictionary of the current state in enough detail to recreate the current state"""
		raise Exception('AbstractRestorableObject::get_state() not implemented')

	def restore_state(self, state: dict):
		"""Restore state from the provided dictionary"""
		raise Exception('AbstractRestorableObject::restore_state() not implemented')
