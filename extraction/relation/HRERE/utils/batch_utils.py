import numpy as np

class Batch_Loader(object):
	def __init__(self, train_triples, n_entities, batch_size=100, neg_ratio=0, contiguous_sampling=False):
		self.train_triples = train_triples
		self.batch_size = batch_size
		self.n_entities = n_entities
		self.contiguous_sampling = contiguous_sampling
		self.neg_ratio = neg_ratio
		self.idx = 0

		self.new_triples = np.empty((self.batch_size * (self.neg_ratio + 1), 3)).astype(np.int64)
		self.new_labels = np.empty((self.batch_size * (self.neg_ratio + 1))).astype(np.float32)
	
	def __call__(self):
		if self.contiguous_sampling:
			if self.idx >= len(self.train_triples):
				self.idx = 0

			b = self.idx
			e = self.idx + self.batch_size
			this_batch_size = len(self.train_triples[b:e])
			self.new_triples[:this_batch_size,:] = self.train_triples[b:e,:]
			self.new_labels[:this_batch_size] = 1.0

			self.idx += this_batch_size
			last_idx = this_batch_size
		else:
			idxs = np.random.randint(0, len(self.train_triples), self.batch_size)
			self.new_triples[:self.batch_size,:] = self.train_triples[idxs,:]
			self.new_labels[:self.batch_size] = 1.0

			last_idx = self.batch_size

		if self.neg_ratio > 0:
			rdm_entities = np.random.randint(0, self.n_entities, last_idx * self.neg_ratio)
			rdm_choices = np.random.random(last_idx * self.neg_ratio)
			self.new_triples[last_idx:(last_idx*(self.neg_ratio+1)),:] = np.tile(self.new_triples[:last_idx,:], (self.neg_ratio, 1))
			self.new_labels[last_idx:(last_idx*(self.neg_ratio+1))] = np.tile(self.new_labels[:last_idx], self.neg_ratio)

			for i in range(last_idx):
				for j in range(self.neg_ratio):
					cur_idx = i * self.neg_ratio + j
					if rdm_choices[cur_idx] < 0.5:
						self.new_triples[last_idx + cur_idx, 0] = rdm_entities[cur_idx]
					else:
						self.new_triples[last_idx + cur_idx, 2] = rdm_entities[cur_idx]
					self.new_labels[last_idx + cur_idx] = -1
			last_idx += cur_idx + 1

		train = {
			"heads": self.new_triples[:last_idx,0], 
			"relations": self.new_triples[:last_idx,1],
			"tails": self.new_triples[:last_idx,2], 
			"labels": self.new_labels[:last_idx]
		}
		return train

class Extended_Batch_Loader(object):
	def __init__(self, train_triples, n_entities, n_relations, batch_size=100, neg_ratio=0, contiguous_sampling=False):
		self.train_triples = train_triples
		self.batch_size = batch_size
		self.n_entities = n_entities
		self.n_relations = n_relations
		self.contiguous_sampling = contiguous_sampling
		self.neg_ratio = neg_ratio
		self.idx = 0

		self.new_triples = np.empty((self.batch_size * (self.neg_ratio*2 + 1), 3)).astype(np.int64)
		self.new_labels = np.empty((self.batch_size * (self.neg_ratio*2 + 1))).astype(np.float32)
	
	def __call__(self):
		if self.contiguous_sampling:
			if self.idx >= len(self.train_triples):
				self.idx = 0

			b = self.idx
			e = self.idx + self.batch_size
			this_batch_size = len(self.train_triples[b:e])
			self.new_triples[:this_batch_size,:] = self.train_triples[b:e,:]
			self.new_labels[:this_batch_size] = 1.0

			self.idx += this_batch_size
			last_idx = this_batch_size
		else:
			idxs = np.random.randint(0, len(self.train_triples), self.batch_size)
			self.new_triples[:self.batch_size,:] = self.train_triples[idxs,:]
			self.new_labels[:self.batch_size] = 1.0

			last_idx = self.batch_size

		if self.neg_ratio > 0:
			rdm_entities = np.random.randint(0, self.n_entities, last_idx * self.neg_ratio)
			rdm_relations = np.random.randint(0, self.n_relations, last_idx * self.neg_ratio)
			rdm_choices = np.random.random(last_idx * self.neg_ratio)
			self.new_triples[last_idx:(last_idx*(self.neg_ratio*2+1)),:] = np.tile(self.new_triples[:last_idx,:], (self.neg_ratio*2, 1))
			self.new_labels[last_idx:(last_idx*(self.neg_ratio*2+1))] = np.tile(self.new_labels[:last_idx], self.neg_ratio*2)

			for i in range(last_idx):
				for j in range(self.neg_ratio):
					cur_idx = i * self.neg_ratio + j
					if rdm_choices[cur_idx] < 0.5:
						self.new_triples[last_idx + cur_idx, 0] = rdm_entities[cur_idx]
					else:
						self.new_triples[last_idx + cur_idx, 2] = rdm_entities[cur_idx]
					self.new_labels[last_idx + cur_idx] = -1
			offset = cur_idx + 1
			for i in range(last_idx):
				for j in range(self.neg_ratio):
					cur_idx = i * self.neg_ratio + j
					self.new_triples[last_idx + offset + cur_idx, 1] = rdm_relations[cur_idx]
					self.new_labels[last_idx + offset + cur_idx] = -1

			last_idx += offset + cur_idx + 1

		train = {
			"heads": self.new_triples[:last_idx,0], 
			"relations": self.new_triples[:last_idx,1],
			"tails": self.new_triples[:last_idx,2], 
			"labels": self.new_labels[:last_idx]
		}
		return train

