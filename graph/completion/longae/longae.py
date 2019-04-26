from graph.completion.graph_completion import GraphCompletion

import os, sys

import numpy as np
from keras import backend as K
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler

from graph.completion.longae.utils_gcn import load_citation_data_from_file, split_citation_data
from graph.completion.longae.utils import compute_masked_accuracy, generate_data, batch_data
from graph.completion.longae.models.ae import autoencoder_multitask
from graph.completion.longae.hparams import hparams

# Benchmarks and metrics: https://github.com/twiet/Knowledge-Graph-Completion/blob/master/README.md
# Parent class: https://github.com/twiet/Knowledge-Graph-Completion/blob/master/graph_completion.py
class longae(GraphCompletion):
	"""longae is the implementation graph_completion for the "Autoencoders for Unsupervised Link Prediction and Semi-Supervised Node Classification"
	project.
	"""	
	def read_dataset(self, fileName, options={}):
		"""
		Reads dataset CORA containing files:
		ind.cora.allx, ind.cora.ally, ind.cora.graph, ind.cora.tx, ind.cora.x, ind.cora.ty, ind.cora.y

		Args:
			fileName: Names of files representing the dataset to read
			options: object to store any extra or implementation specific data

		Returns:
			Graph adjacency matrix, node features matrix, node labels matrix
		"""		
		adj, feats, y_train, y_val, y_test, mask_train, mask_val, mask_test = load_citation_data_from_file(fileName)

		feats = MaxAbsScaler().fit_transform(feats)
		train = adj.copy()

		test_inds = split_citation_data(adj)
		test_inds = np.vstack({tuple(row) for row in test_inds})
		test_r = test_inds[:, 0]
		test_c = test_inds[:, 1]
		labels = []
		labels.extend(np.squeeze(adj[test_r, test_c].toarray()))
		labels.extend(np.squeeze(adj[test_c, test_r].toarray()))

		# If multitask, simultaneously perform link prediction and
		# semi-supervised node classification on incomplete graph with
		# 10% held-out positive links and same number of negative links.
		# If not multitask, perform node classification with complete graph.
		train[test_r, test_c] = -1.0
		train[test_c, test_r] = -1.0
		adj[test_r, test_c] = 0.0
		adj[test_c, test_r] = 0.0

		adj.setdiag(1.0)
		train.setdiag(1.0)

		print('\nCompiling autoencoder model...\n')
		self.encoder, self.ae = autoencoder_multitask(adj, feats, y_train)
		adj = sp.hstack([adj, feats]).tolil()
		train = sp.hstack([train, feats]).tolil()
		print(self.ae.summary())
		
		train_data = {
			"train": train,
			"adj": adj,
			"feats": feats,
			"y_train": y_train,
			"mask_train": mask_train
		}

		validation_data = {
			"test_r": test_r,
			"test_c": test_c,
			"adj": adj,
			"feats": feats,
			"y_true": y_val,
			"mask_true": mask_val,
			"labels": labels
		}

		test_data = {
			"test_r": test_r,
			"test_c": test_c,
			"adj": adj,
			"feats": feats,
			"y_true": y_test,
			"mask_true": mask_test,
			"labels": labels
		}

		return train_data, validation_data, test_data

	def train(self, data, options={}):
		"""
		Trains AE model base on adjacency and/or feature matrix

		Args:
			data: sparse matrix contain node to node relationship and feature data
			options: object to store any extra or implementation specific data

		Returns:
			ret: None. Trained model stored internally to instance's state. 
		"""
		train = data["train"]
		adj = data["adj"]
		feats = data["feats"]
		y_train = data["y_train"]
		mask_train = data["mask_train"]

		# validation params
		y_val = options["y_true"]
		mask_val = options["mask_true"]
		labels = options["labels"]
		test_r = options["test_r"]
		test_c = options["test_c"]

		train_data = generate_data(adj, train, feats, y_train, mask_train, shuffle=True)
		print('\nFitting autoencoder model...\n')

		batched_data = batch_data(train_data, hparams.train_batch_size)
		num_iters_per_train_epoch = adj.shape[0] / hparams.train_batch_size
		scores = [[], [], [], []]
		
		for e in range(hparams.epochs):
			print('\nEpoch {:d}/{:d}'.format(e+1, hparams.epochs))
			print('Learning rate: {:6f}'.format(K.eval(self.ae.optimizer.lr)))
			curr_iter = 0
			train_loss = []
			for batch_a, batch_t, batch_f, batch_y, batch_m in batched_data:
				# Each iteration/loop is a batch of train_batch_size samples
				batch_y = np.concatenate([batch_y, batch_m], axis=1)
				res = self.ae.train_on_batch([batch_a, batch_f], [batch_t, batch_y])
				train_loss.append(np.mean(res))
				curr_iter += 1
				if curr_iter >= num_iters_per_train_epoch:
					break
			train_loss = np.asarray(train_loss)
			train_loss = np.mean(train_loss, axis=0)
			print('Avg. training loss: {:s}'.format(str(train_loss)))

			# Training Evaluation
			from sklearn.metrics import roc_auc_score as auc_score
			from sklearn.metrics import average_precision_score as ap_score

			metrics = {
				"auc_score": auc_score,
				"ap_score": ap_score
			}

			prediction_data = self.predict({
				"adj": adj,
				"feats": feats
			})
			
			auc, ap, acc = self.evaluate({
				"y_true": y_val,
				"mask_true": mask_val,
				"labels": labels,
				"test_r": test_r,
				"test_c": test_c
			}, metrics, prediction_data)
			
			scores[0].append(train_loss)
			scores[1].append(auc)
			scores[2].append(ap)
			scores[3].append(acc)

			if e % 50 == 0:
				self.save_model(hparams.checkpoint_dir + "checkpoint_{}.h5".format(e))
		# self.__save_fig("scores.png", scores)
		print('\nAll done.')
		return scores
	
	def predict(self, data, options={}):
		"""
		Predicts links on the given input data (e.g. knowledge graph). Assumes model has been trained with train()

		Args:
			data: sparse matrix with features containing missing links 
			options: object to store any extra or implementation specific data

		Returns:
			predictions: [tuple,...], i.e. list of predicted tuples. 
				Each tuple likely will follow format: (subject_entity, relation, object_entity), but isn't required.
		"""
		adj = data["adj"]
		feats = data["feats"]

		print('\nEvaluating validation set...')
		lp_scores, nc_scores = [], []
		for step in range(adj.shape[0] // hparams.val_batch_size + 1):
			low = step * hparams.val_batch_size
			high = low + hparams.val_batch_size
			batch_adj = adj[low:high].toarray()
			batch_feats = feats[low:high]
			if batch_adj.shape[0] == 0:
				break
			decoded = self.ae.predict_on_batch([batch_adj, batch_feats])
			decoded_lp = decoded[0] # link prediction scores
			decoded_nc = decoded[1] # node classification scores
			lp_scores.append(decoded_lp)
			nc_scores.append(decoded_nc)
		lp_scores = np.vstack(lp_scores)
		nc_scores = np.vstack(nc_scores)
		return {
			"lp_scores": lp_scores, 
			"nc_scores": nc_scores
		}

	def evaluate(self, benchmark_data, metrics={}, options={}):
		"""
		Calculates evaluation metrics on chosen benchmark dataset.
		Precondition: model has been trained and predictions were generated from predict()

		Args:
			benchmark_data: Longae project only works on CORA dataset
			metrics: Area under curve (AUC) and Average Precision (AP)
			options: object to store any extra or implementation specific data

		Returns:
			evaluations: dictionary of scores with respect to chosen metrics
		"""
		lp_scores = options["lp_scores"]
		nc_scores = options["nc_scores"]

		test_r = benchmark_data["test_r"]
		test_c = benchmark_data["test_c"]
		y_true = benchmark_data["y_true"]
		labels = benchmark_data["labels"]
		mask_true = benchmark_data["mask_true"]

		auc_score = metrics["auc_score"]
		ap_score = metrics["ap_score"]

		print('\nEvaluating model')
		predictions = []
		predictions.extend(lp_scores[test_r, test_c])
		predictions.extend(lp_scores[test_c, test_r])
		
		auc = auc_score(labels, predictions)
		ap = ap_score(labels, predictions)
		node_val_acc = compute_masked_accuracy(y_true, nc_scores, mask_true)

		print('Val AUC: {:6f}'.format(auc))
		print('Val AP: {:6f}'.format(ap))
		print('Node Val Acc {:f}'.format(node_val_acc))

		return auc, ap, node_val_acc

	def save_model(self, file):
		"""

		:param file: Where to save the model - Optional function
		:return:
		"""
		self.ae.save_weights(file)

	def load_model(self, file):
		"""

		:param file: From where to load the model - Optional function
		:return:
		"""
		if len(file) == 0:
			return
		
		self.ae.load_weights(file)

	def __save_fig(self, file, scores):
		import matplotlib.pyplot as plt
		
		f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
		plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
		ax1.plot(scores[0])
		ax1.set_xlabel('epoch')
		ax1.set_title("training loss", fontsize=18)
		ax2.plot(scores[1])
		ax2.set_xlabel('epoch')
		ax2.set_title("auc score", fontsize=18)
		ax3.plot(scores[2])
		ax3.set_xlabel('epoch')
		ax3.set_title("ap score", fontsize=18)
		ax4.plot(scores[3])
		ax4.set_xlabel('epoch')
		ax4.set_title("node accuracy", fontsize=18)
		plt.savefig(file)