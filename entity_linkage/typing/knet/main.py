import os
import sys
import numpy as np
import tensorflow as tf
import util
import model as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from entity_typing import entity_typing

class knet():
    def __init__(self):
        self.type_file = None
        self.disamb_file = None
        self.embedding = None
        self.glove = None
        self.train_context = None
        self.train_entity = None
        self.train_fbid = None
        self.train_labels = None
        self.valid_context = None
        self.valid_entity = None
        self.valid_fbid = None
        self.valid_labels = None
        self.model_dir = "./model"
        self.test_file = None
        self.predict_types = None
        self.final_result = None

    def read_dataset(self, file_names, options={}):
        assert len(file_names) >= 12, "Check README and get all the files"
        if os.path.exists(file_names[0]):
            self.type_file = file_names[0]
        else:
            assert False, "type file is not present at the given location"

        if os.path.exists(file_names[1]):
            self.disamb_file = file_names[1]
        else:
            assert False, "disamb file is not present at the given location"

        if os.path.exists(file_names[2]):
            self.embedding = file_names[2]
        else:
            assert False, "embedding file is not present at the given location"

        if os.path.exists(file_names[3]):
            self.glove = file_names[3]
        else:
            assert False, "glove file is not present at the given location"

        if os.path.exists(file_names[4]):
            self.train_context = file_names[4]
            if self.train_context.find(".npy") == -1:
                assert False, "train context file is not a valid numpy file"
        else:
            assert False, "train context file is not present at the given location"

        if os.path.exists(file_names[5]):
            self.train_entity = file_names[5]
            if self.train_entity.find(".npy") == -1:
                assert False, "train entity file is not a valid numpy file"
        else:
            assert False, "train entity file is not present at the given location"

        if os.path.exists(file_names[6]):
            self.train_fbid = file_names[6]
            if self.train_fbid.find(".npy") == -1:
                assert False, "train fbid file is not a valid numpy file"
        else:
            assert False, "train fbid file is not present at the given location"

        if os.path.exists(file_names[7]):
            self.train_labels = file_names[7]
            if self.train_labels.find(".npy") == -1:
                assert False, "train labels file is not a valid numpy file"
        else:
            assert False, "train labels file is not present at the given location"

        if os.path.exists(file_names[8]):
            self.valid_context = file_names[8]
            if self.valid_context.find(".npy") == -1:
                assert False, "valid context file is not a valid numpy file"
        else:
            assert False, "valid context file is not present at the given location"

        if os.path.exists(file_names[9]):
            self.valid_entity = file_names[9]
            if self.valid_entity.find(".npy") == -1:
                assert False, "valid_entity file is not a valid numpy file"
        else:
            assert False, "valid entity file is not present at the given location"

        if os.path.exists(file_names[10]):
            self.valid_fbid = file_names[10]
            if self.valid_fbid.find(".npy") == -1:
                assert False, "valid_fbid file is not a valid numpy file"
        else:
            assert False, "valid fbid file is not present at the given location"

        if os.path.exists(file_names[11]):
            self.valid_labels = file_names[11]
            if self.valid_labels.find(".npy") == -1:
                assert False, "valid_labels file is not a valid numpy file"
        else:
            assert False, "valid labels file is not present at the given location"

    def train(self, train_data=None, options={}):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        util.printlog("Loading Data")
        embedding = np.load(self.embedding)
        train_entity = np.load(self.train_entity)
        train_context = np.load(self.train_context)
        train_label = np.load(self.train_labels)
        train_fbid = np.load(self.train_fbid)

        valid_entity = np.load(self.valid_entity)
        valid_context = np.load(self.valid_context)
        valid_label = np.load(self.valid_labels)
        valid_fbid = np.load(self.valid_fbid)

        train_size = len(train_entity)
        if train_size < 100: 
            batch_size = train_size / 10
            iter_num = train_size
            check_freq = train_size / 10
        elif train_size < 10000:
            batch_size = train_size / 100
            iter_num = train_size / 10
            check_freq = train_size / 100
        else:
            batch_size = train_size / 1000
            iter_num = train_size / 100
            check_freq = train_size / 1000

        batch_size = int(batch_size)
        iter_num = int(iter_num)
        check_freq = int(check_freq)

        model = models.KA_D("KA+D", self.disamb_file)

        sess = tf.Session()
        w2v = util.build_vocab(self.glove, model.word_size)
        sess.run(model.initializer)

        util.printlog("Begin training")

        for i in range(iter_num):
            if i % check_freq == 0:
                util.printlog("Validating after running " + str(int(i * batch_size / train_size)) + " epoches")
                util.test(w2v, model, valid_entity, valid_context, valid_label, valid_fbid, embedding, batch_size, sess,
                          "all")
                model.saver.save(sess, os.path.join(self.model_dir, str(i)))

            fd = model.fdict(w2v, (i * batch_size) % train_size, batch_size, 1, train_entity, train_context,
                             train_label, train_fbid, embedding, False)
            fd[model.kprob] = 0.5
            sess.run(model.train, feed_dict=fd)

            if i % int(train_size / batch_size / 10) == 0:
                util.printlog("Epoch {}, Batch {}".format(int((i * batch_size) / train_size), int((i * batch_size) % train_size / batch_size)))

    def predict(self, test_data, model_details=None, options={}):
        assert len(test_data) != 0, "test_data list shouldn't be empty"
        self.test_file = test_data[0]
        if not os.path.exists(self.test_file):
            assert False, "File doesn't exists"

        direct_entity, direct_context, self.predict_types = util.raw2npy(self.test_file)

        embedding = np.load(self.embedding)
        model = models.KA_D("KA+D", self.disamb_file)

        sess = tf.Session()
        w2v = util.build_vocab(self.glove, model.word_size)
        sess.run(model.initializer)
        util.printlog("Begin computing direct outputs")
        self.final_result = util.direct(w2v, sess, model, direct_entity, direct_context, embedding, self.type_file)

        dir_name = os.path.dirname(test_data[0])
        output_file = os.path.join(dir_name, "output_knet.txt")
        final_str = ""
        for i in self.final_result:
            final_str = "{}\n{}".format(final_str, i)
        with open(output_file, 'w') as fin:
            fin.write(final_str.strip())
        print("Result has been output to following file: {}".format(output_file))
        return self.final_result

    def evaluate(self, test_data, prediction_data=None, options={}):
        if options.get("paper", None) is not None:
            assert len(test_data) >= 8, "All files are not given"
            
            embedding = np.load(self.embedding)
            test_context = np.load(test_data[0])
            test_entity = np.load(test_data[1])
            test_fbid = np.load(test_data[2])
            test_label = np.load(test_data[3])

            manual_context = np.load(test_data[4])
            manual_entity = np.load(test_data[5])
            manual_fbid = np.load(test_data[6])
            manual_label = np.load(test_data[7])

            model = models.KA_D("KA+D", self.disamb_file)

            sess = tf.Session()
            w2v = util.build_vocab(self.glove, model.word_size)
            sess.run(model.initializer)

            batch_size = 1000

            util.printlog("Test on the wiki-auto test set")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "all")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "succ")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "miss")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "person")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "organization")
            util.test(w2v, model, test_entity, test_context, test_label, test_fbid,
                embedding, batch_size, sess, "location")

            util.printlog("Test on the wiki-man test set")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "all")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "succ")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "miss")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "person")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "organization")
            util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid,
                embedding, batch_manual, sess, "location")
            return
        if len(test_data) > 0 and os.path.exists(self.test_file):
            _ = self.predict(test_data)

        precision, recall, f1_score = util.calculate_precision_recall(self.predict_types, self.final_result)
        return (precision, recall, f1_score)


    def save_model(self, file):
        """

        :param file: Where to save the model - Optional function
        :return:
        """
        pass

    def load_model(self, file):
        """

        :param file: From where to load the model - Optional function
        :return:
        """
        pass


if '__main__' == __name__:
    knet_instance = knet()
    knet_instance.read_dataset([
        "data/types",
        "data/disamb_file",
        "data/embedding.npy",
        "data/sample_glove.txt",
        "data/sample_train_context.npy",
        "data/sample_train_entity.npy",
        "data/sample_train_fbid.npy",
        "data/sample_train_label.npy",
        "data/sample_valid_context.npy",
        "data/sample_valid_entity.npy",
        "data/sample_valid_fbid.npy",
        "data/sample_valid_label.npy"
    ])

    knet_instance.train(None)
    knet_instance.predict(["data/entity_typing_test_input.txt"])
    knet_instance.evaluate([])
    knet_instance.evaluate([
        "data/test_context",
        "data/test_entity",
        "data/test_fbid.npy",
        "data/test_label.txt",
        "data/manual_context.npy",
        "data/manual_entity.npy",
        "data/manual_fbid.npy",
        "data/manual_label.npy"
    ], options={"paper": True})