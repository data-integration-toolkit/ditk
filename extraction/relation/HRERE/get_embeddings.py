import tensorflow as tf
from optparse import OptionParser
import numpy as np
import os
import config
from utils.data_utils import load_dict_from_txt

def parse_args(parser):
    parser.add_option("-e", "--embed", dest="embed_type", type="string")
    parser.add_option("-m", "--model", dest="model_name", type="string")
    parser.add_option("-o", "--output", dest="output_path", type="string")
    options, args = parser.parse_args()
    return options, args

def get_real_embeddings(model_name, output_path):
    checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        entity = graph.get_tensor_by_name("entity_embedding:0")
        relation = graph.get_tensor_by_name("relation_embedding:0")
        e, r = sess.run([entity, relation])
        np.save(os.path.join(output_path, "entity.npy"), e)
        np.save(os.path.join(output_path, "relation.npy"), r)

def get_complex_embeddings(model_name, output_path):
    checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        entity1 = graph.get_tensor_by_name("entity_embedding1:0")
        entity2 = graph.get_tensor_by_name("entity_embedding2:0")
        relation1 = graph.get_tensor_by_name("relation_embedding1:0")
        relation2 = graph.get_tensor_by_name("relation_embedding2:0")
        e1, e2, r1, r2 = sess.run([entity1, entity2, relation1, relation2])
        np.save(os.path.join(output_path, "entity1.npy"), e1)
        np.save(os.path.join(output_path, "entity2.npy"), e2)
        np.save(os.path.join(output_path, "relation1.npy"), r1)
        np.save(os.path.join(output_path, "relation2.npy"), r2)

def get_complex_scores(model_name, output_path):
    checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        heads = graph.get_operation_by_name("head_entities").outputs[0]
        tails = graph.get_operation_by_name("tail_entities").outputs[0]
        relations = graph.get_operation_by_name("relations").outputs[0]
        pred = graph.get_operation_by_name("pred").outputs[0]

        relation2id = load_dict_from_txt("../HRERE/data/relation2id.txt")
        id2r = {relation2id[x]: x for x in relation2id.keys()}
        r2id = load_dict_from_txt(config.FB3M_R2ID)
        r = []
        for i in range(1, 55):
            r.append(r2id[id2r[i]])
        infile = open(config.FB1M_TEST)
        cnt = 0
        for line in infile.readlines():
            e1, l, e2 = line.strip().split(",")
            e1, l, e2 = int(e1), int(l), int(e2)
            res = sess.run(pred, feed_dict={heads: [e1] * 54,
                tails: [e2] * 54, relations: r})
            p = np.argmax(res)
            if r[p] == l:
                cnt += 1
        print(cnt)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    if options.embed_type == "real":
        get_real_embeddings(options.model_name, options.output_path)
    elif options.embed_type == "complex":
        get_complex_embeddings(options.model_name, options.output_path)
    else:
        get_complex_scores(options.model_name, options.output_path)
