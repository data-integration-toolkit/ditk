import tensorflow as tf
import numpy as np
from utils.batch_utils import Batch_Loader
from utils.data_utils import load_dict_from_txt
from optparse import OptionParser
import config
import os

def parse_args(parser):
	parser.add_option("-d", "--data_name", default="fb15k", dest="data_name", type="string")
	parser.add_option("-a", "--head", default=-1, dest="head", type="int")
	parser.add_option("-b", "--tail", default=-1, dest="tail", type="int")
	parser.add_option("-r", "--relation", default=-1, dest="relation", type="int")

	options, args = parser.parse_args()
	return options, args

def main(options):
	if options.data_name == "fb15k":
		e2id = load_dict_from_txt(config.FB15K_E2ID)
		r2id = load_dict_from_txt(config.FB15K_R2ID)
		n_entities = len(e2id)
		i2e = {v:k for k, v in e2id.items()}
		i2r = {v:k for k, v in r2id.items()}
	e1 = options.head
	e2 = options.tail
	r = options.relation
	if r == -1:
		raise AttributeError("Please specify the relation!")
	if (e1 == -1) and (e2 == -1):
		raise AttributeError("Please specify one entity!")
	if (e1 != -1) and (e2 != -1):
		raise AttributeError("Please specify only one entity!")
	idx_mat = np.empty((n_entities, 3), dtype=np.int64)
	if e1 == -1:
		idx_mat[:,1:] = np.tile((r,e2), (n_entities,1))
		idx_mat[:,0] = np.arange(n_entities)
	else:
		idx_mat[:,:2] = np.tile((e1,r), (n_entities,1))
		idx_mat[:,2] = np.arange(n_entities)

	checkpoint_file = os.path.join(config.CHECKPOINT_PATH, "best_Complex_tanh_fb15k")
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		heads = graph.get_operation_by_name("head_entities").outputs[0]
		tails = graph.get_operation_by_name("tail_entities").outputs[0]
		relations = graph.get_operation_by_name("relations").outputs[0]
		labels = graph.get_operation_by_name("labels").outputs[0]

		pred = graph.get_operation_by_name("pred").outputs[0]

		preds = sess.run(pred, {heads: idx_mat[:,0], tails: idx_mat[:,2], relations: idx_mat[:,1]})
		scores = {x:y for x,y in enumerate(preds)}
		cnt = 0
		print("Top 10 Candidates for (%s, %s, %s):" % (i2e.get(e1, "_"), i2r[r], i2e.get(e2, "_")))
		for w in sorted(scores, key=scores.get, reverse=True):
			cnt += 1
			if cnt > 10:
				break
			print(i2e[w], scores[w])


if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
