#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from simplE_ignr import *

from reader import *
from tensor_factorizer import *
from params import *

class TrainerTester:
	def __init__(self, model_name, params, dataset, data=None, reader=None):
		instance_gen = globals()[model_name]
		self.model = instance_gen(params=params, dataset=dataset)
		self.model.setup_reader(data, reader)

	def train(self):
		self.model.setup_weights()
		self.model.setup_saver()
		self.model.create_train_placeholders()
		self.model.gather_train_embeddings()
		self.model.create_train_model()
		self.model.define_regularization()
		self.model.create_optimizer()
		self.model.create_session()
		self.model.optimize()
		self.model.close_session()

	def test_model_on(self, itr, valid_or_test="test"):
		tf.reset_default_graph()
		self.model.setup_weights()
		self.model.setup_loader()
		self.model.create_test_placeholders()
		self.model.gather_test_embeddings()
		self.model.create_test_model()
		self.model.create_session()
		self.model.load_session(itr)
		raw_mrr, raw_hit1, raw_hit3, raw_hit10, fil_mrr, fil_hit1, fil_hit3, fil_hit10, new_triples = self.model.test(self.model.reader.triples[valid_or_test])
		# print(raw_mrr, raw_hit1, raw_hit3, raw_hit10, fil_mrr, fil_hit1, fil_hit3, fil_hit10)
		return {
			"mrr": fil_mrr,
			"hits1": fil_hit1,
			"hits3": fil_hit3,
			"hits10": fil_hit10
		}, new_triples

	def early_stop(self):
		best_mrr, best_itr = -1, "-1"
		
		for itr in self.model.params.get_early_stopping_itrs():
			print("Early Stop Iteration", itr)
			metrics, triples = self.test_model_on(itr=itr)
			if metrics["mrr"] > best_mrr:
				best_mrr, best_itr = metrics["mrr"], itr

		self.model.close_session()
		print("Best Iteration:", best_itr)
		return best_itr, best_mrr
		
	def test(self, itr):
		self.test_model_on(itr=itr, valid_or_test="test")
		self.model.close_session()

	def train_earlystop_test(self):
		print("Training " + self.model.model_name + " on " + self.model.dataset + " with emb_size = " + str(self.model.params.emb_size) + ", learning rate = " + str(self.model.params.learning_rate) + " neg_ratio = " + str(self.model.params.neg_ratio) + " alpha = " + str(self.model.params.alpha))
		self.train()

		print("Early stoppin for " + self.model.model_name + " on " + self.model.dataset + " with emb_size = " + str(self.model.params.emb_size) + ", itrs = ", self.model.params.get_early_stopping_itrs())
		best_itr, _ = self.early_stop()

		print("Testing " + self.model.model_name + " on " + self.model.dataset + " with emb_size = " + str(self.model.params.emb_size) + ", best itr = " + best_itr)
		self.test(itr=best_itr)


	