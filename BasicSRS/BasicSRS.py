#!/usr/bin/python3

import os
import time
import math

from tkinter import *

from numpy.random import random as npr

controls = {'[MASK]', '[SEG]'}


def print_delay(start, finish):
	months = (finish-start)//(86400*28)
	days = (finish-start)//86400 - 28*months
	hours = (finish-start)//3600 - 24*28*months - 24*days

	if months:
		return str(months)+' months, '+str(days)+' days, and '+str(hours)+' hours'
	if days:
		return str(days)+' days and '+str(hours)+' hours'
	return str(hours)+' hours'


class BasicSRS(Tk):
	"""GUI class for reviewing flashcards :D"""

	def prep_tokens(self):
		"""There is no 'tokens' file, so one must be created.
		To create the 'tokens' file, it is assumed that the learner knows 90% of vocab by frequency"""

		assumed_known = 0.90

		tokens = open('tokens.db', 'w', encoding='utf-8')
		cumulative = 0
		for freq, key in list(reversed(sorted([(freq, key) for key, freq in self.frequency.items()]))):
			cumulative += freq
			if cumulative < assumed_known:
				tokens.write(key + '\t' + str(int(time.time())) + '\t' + str(int(self.schedule_new(4))) + '\n')
		tokens.close()

	def prep_cache(self):
		"""The 'contexts' file can be seriously huge, so I keep a cache of i+1 examples on-hand.
		This cache consists of example contexts in which only one or two tokens are unknown."""

		knowns = set(self.tokens.keys())

		cache = []
		with open('contexts.db', 'r', encoding='utf-8') as f:
			for line in f:
				key = line.split('\t')[0]
				tok = [key] + [x for x in line.split('\t')[1].split() if x not in controls]

				if key in knowns:
					cache.append(line)
				elif len(set(tok)-knowns) <= 2:
					cache.append(line)

		open('cache.db', 'w', encoding='utf-8').write(''.join(cache))

	def get_card_score(self, knowns, tok, key):
		"""Estimate the difficulty of a flashcard, using inverse frequency of occurrence as a proxy."""

		if key in self.tokens:
			return 0

		if len(set(tok)-knowns):
			unknowns = sum([math.log(self.frequency[x])/math.log(2) for x in set(tok)-knowns])
			return int(len(set(tok)-knowns) * (2**-(unknowns/len(set(tok)-knowns))))
		else:
			return 0

	def save(self):
		save = open('tokens.db', 'a', encoding='utf-8')
		for _, tup in self.scores.items():
			key = tup[1][0]
			rep = tup[1][1]
			due = tup[1][2]
			save.write(key + '\t' + str(rep) + '\t' + str(due) + '\n')
		save.close()

	def load_edict(self):
		self.edict = dict()
		if os.path.exists('edict.db'):
			for entry in [x for x in open('edict.db', 'r', encoding='utf-8').read().split('\n') if len(x)]:
				key = entry.split('/')[0].split('[')[0].replace(' ', '')
				self.edict[key] = entry

	def load(self):
		"""Load files needed to carry out functions related to reviewing flashcards."""

		self.frequency = {key: float(freq) for key, freq in [x.split() for x in open('frequency.db', 'r', encoding='utf-8').read().split('\n') if len(x)]}

		if not os.path.exists('tokens.db'):
			print('tokens.db does not exist -- creating tokens.db')
			self.prep_tokens()

		self.tokens = {key: (float(rep), float(due)) for key, rep, due in [x.split() for x in open('tokens.db', 'r', encoding='utf-8').read().split('\n') if len(x)]}
		print(len(self.tokens), 'tokens loaded')

		if not os.path.exists('cache.db'):
			print('cache.db does not exist -- creating cache.db')
			self.prep_cache()

		self.flashcards = self.arrange_flashcards([x.split('\t') for x in open('cache.db', 'r', encoding='utf-8').read().split('\n') if len(x)])
		print(len(self.flashcards), 'flashcards loaded')

		self.scores = dict()

		self.load_edict()

	def arrange_flashcards(self, contexts):
		"""Score flashcards so they can be shown in order of increasing difficulty."""

		# first, check to see if we have any flashcards that are due right now
		candidates = dict()

		due = set([key for key, tup in self.tokens.items() if tup[1] < time.time()])
		for fc in contexts:
			if fc[0] in due:
				if fc[0] not in candidates:
					candidates[fc[0]] = []
				candidates[fc[0]].append(fc)

		knowns = set(self.tokens.keys())

		# next, check to see how many i+1 flashcards we have
		for fc in contexts:
			if fc[0] not in knowns and len(set([x for x in fc[1].split() if x not in controls])-knowns) == 0:
				if fc[0] not in candidates:
					candidates[fc[0]] = []
				candidates[fc[0]].append(fc)

		flashcards = []
		for key, collection in candidates.items():
			selection = collection[int(len(collection)*npr())]
			tokens = [x for x in selection[1].split()]
			flashcards.append((self.get_card_score(knowns.union(controls), set([key]+tokens), key), key, tokens))

		return sorted(flashcards)

	@staticmethod
	def schedule_new(level):
		"""Assign a delay based on the level of difficulty."""
		return int(time.time() + 86400 * (5**level) * (0.90 + npr()*0.20))

	def schedule(self, token, passed):
		"""Assign a delay based on whether it's a pass or fail unless a score is provided."""

		if token not in self.tokens:
			return self.schedule_new(0)

		rep_due = self.tokens[token]
		exp = math.log(rep_due[1]-rep_due[0])
		if passed:
			return int(time.time() + (math.e**(exp+1)) * (0.90 + npr()*0.20))
		else:
			return int(time.time() + (math.e**max(0, exp-1)) * (0.90 + npr()*0.20))

	def redraw_and_clear(self):
		"""Update text fields to show the current state of things."""

		self.noteview.delete(1.0, END)
		self.listview.delete(1.0, END)

		if self.analysis_mode:
			base, mask = 'base', 'mask'
			keypos = [x for x in range(len(self.flashcards[self.selection][2])) if self.flashcards[self.selection][2][x] == '[MASK]'][0]
			self.noteview.insert(END, ''.join([x for x in self.flashcards[self.selection][2][:keypos] if x not in controls]), base)
			self.noteview.insert(END, self.flashcards[self.selection][1], mask)
			self.noteview.insert(END, ''.join([x for x in self.flashcards[self.selection][2][keypos+1:] if x not in controls]), base)
			self.noteview.insert(END, '\n')

			key = self.flashcards[self.selection][1]
			tokens = [x for x in self.flashcards[self.selection][2] if x != '[SEG]']
			for tok in [x if x != '[MASK]' else key for x in tokens]:
				self.listview.insert(END, tok)
				if tok in self.edict:
					self.listview.insert(END, '\t' + self.edict[tok] + '\n')
				else:
					self.listview.insert(END, '\n')
		else:
			for i in range(max(0, self.selection-10), min(len(self.flashcards), self.selection+100)):

				key = self.flashcards[i][1]
				base, mask = 'base', 'mask'
				if key in self.tokens:
					base, mask = 'due_base', 'due_mask'
				else:
					base, mask = 'new_base', 'new_mask'

				if i == self.selection:
					base, mask = 'selected_base', 'selected_mask'
					if self.flashcards[i][1] in self.edict:
						self.noteview.insert(END, self.edict[self.flashcards[i][1]], 'base')
					else:
						self.noteview.insert(END, 'no entry available')

				self.listview.insert(END, str(self.flashcards[i][0])+'\t', base)
				if i in self.scores:
					result, (word, rep_date, due_date) = self.scores[i]
					self.listview.insert(END, word + ': ' + result + ' - due in ' + print_delay(rep_date, due_date), base)
					self.listview.insert(END, '\n')
				else:
					keypos = [x for x in range(len(self.flashcards[i][2])) if self.flashcards[i][2][x] == '[MASK]'][0]
					self.listview.insert(END, ''.join([x for x in self.flashcards[i][2][:keypos] if x not in controls]), base)
					self.listview.insert(END, self.flashcards[i][1], mask)
					self.listview.insert(END, ''.join([x for x in self.flashcards[i][2][keypos+1:] if x not in controls]), base)
					self.listview.insert(END, '\n')

	def keyboard(self, event):

		# navigation
		if event.keysym == 'Up':
			self.selection -= 1
		if event.keysym == 'Down':
			self.selection += 1
		if event.keysym == 'Prior':
			self.selection -= 30
		if event.keysym == 'Next':
			self.selection += 30

		# scoring
		if event.keysym == 'Left':
			self.scores[self.selection] = ('Fail', (self.flashcards[self.selection][1], int(time.time()), self.schedule(self.flashcards[self.selection][1], False)))
		if event.keysym == 'Right':
			self.scores[self.selection] = ('Pass', (self.flashcards[self.selection][1], int(time.time()), self.schedule(self.flashcards[self.selection][1], True)))

		if event.keysym == '1':
			self.scores[self.selection] = ('Init 1', (self.flashcards[self.selection][1], int(time.time()), self.schedule_new(1)))
		if event.keysym == '2':
			self.scores[self.selection] = ('Init 2', (self.flashcards[self.selection][1], int(time.time()), self.schedule_new(2)))
		if event.keysym == '3':
			self.scores[self.selection] = ('Init 3', (self.flashcards[self.selection][1], int(time.time()), self.schedule_new(3)))
		if event.keysym == '4':
			self.scores[self.selection] = ('Init 4', (self.flashcards[self.selection][1], int(time.time()), self.schedule_new(4)))
		if event.keysym == '6':
			self.scores[self.selection] = ('Banish', (self.flashcards[self.selection][1], int(time.time()), self.schedule_new(6)))

		if event.keysym == 'space':
			if self.selection in self.scores:
				self.scores.pop(self.selection)
		if event.keysym == 'Return':
			self.analysis_mode ^= 1

		# save/recalc
		if event.keysym == 'l':
			self.load()
		if event.keysym == 's':
			self.save()
			self.load()
		if event.keysym == 'r':
			self.save()
			self.prep_cache()
			self.load()

		if self.selection < 0:
			self.selection = 0
		if self.selection >= len(self.flashcards):
			self.selection = len(self.flashcards)-1

		# print (event.keysym, self.selection)

		self.redraw_and_clear()

	def __init__(self):
		Tk.__init__(self)
		self.minsize(500, 300)

		self.noteview = Text(self, height=5, background='white')
		self.noteview.pack(side=TOP, fill=BOTH, expand=0)

		self.noteview.tag_configure('base', foreground='#000000', font=('Mincho', 12))
		self.noteview.tag_configure('mask', foreground='#ff0000', font=('Mincho', 12))

		self.listview = Text(self, wrap='none', height=20, background='white')
		self.listview.pack(side=TOP, fill=BOTH, expand=1)

		self.listview.tag_configure('base', foreground='#000000', background='#eeeeee', font=('Mincho', 12))
		self.listview.tag_configure('mask', foreground='#ff0000', background='#eeeeee', font=('Mincho', 12))
		self.listview.tag_configure('due_base', foreground='#000000', background='#ffeeee', font=('Mincho', 12))
		self.listview.tag_configure('due_mask', foreground='#ff0000', background='#ffeeee', font=('Mincho', 12))
		self.listview.tag_configure('new_base', foreground='#000000', background='#eeffee', font=('Mincho', 12))
		self.listview.tag_configure('new_mask', foreground='#ff0000', background='#eeffee', font=('Mincho', 12))
		self.listview.tag_configure('selected_base', foreground='#000000', background='#eeeeff', font=('Mincho', 12))
		self.listview.tag_configure('selected_mask', foreground='#ff0000', background='#eeeeff', font=('Mincho', 12))

		self.bind_all('<KeyPress>', self.keyboard)

		self.frequency = dict()
		self.tokens = dict()
		self.scores = dict()
		self.edict = dict()
		self.flashcards = []
		self.selection = 0
		self.analysis_mode = 0

		self.load()

		self.redraw_and_clear()


BasicSRS().mainloop()
