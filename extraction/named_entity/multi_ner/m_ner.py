import collections
import os
import re
from mner import modeling
from mner import optimization
from mner import tokenization
import tensorflow as tf
import tf_metrics
import pickle

from configure import FLAGS

from ner import Ner


class InputExample(object):

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_data_from_list(self, data_list):
        lines = []
        words = []
        labels = []
        for i in range(len(data_list)):
            if len(data_list[i]) == 0:
                if len(words) and words[-1] == '.':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                continue
            if data_list[i][0].startswith("-DOCSTART-"):
                words.append('')
                continue
            words.append(data_list[i][0])
            labels.append(data_list[i][-1])
        return lines

    def _read_data(self, input_file):
        """Reads a BIO data."""
        lines = [line.strip().split(' ') for line in open(input_file)]
        return self._read_data_from_list(lines)


class NerProcessor(DataProcessor):
    def ditk_ner_to_conll2003(self, lines):
        def is_chunk_tag(chunk):
            return chunk == 'O' or str(chunk).startswith('B-') or str(chunk).startswith('I-')

        conll_2000_tagset = {'NP', 'ADVP', 'ADJP', 'VP', 'PP', 'SBAR', 'CONJP', 'PRT', 'INTJ', 'LST', 'UCP'}
        converted_lines = []

        chunk_tag_format = None
        for i, line in enumerate(lines):
            if len(line.strip()) > 0:
                data = line.split()
                if chunk_tag_format is None:
                    chunk_tag_format = is_chunk_tag(data[2])

                if not chunk_tag_format:
                    tokens = str(data[2])
                    tokens = re.findall(r"[\w']+", tokens)
                    for token in tokens:
                        if token in conll_2000_tagset:
                            data[2] = 'I-'+token
                        else:
                            data[2] = 'O'

                converted_line = list()
                converted_line.extend(data[0:4])
                converted_lines.append(converted_line)
            else:
                converted_lines.append(line)
        return converted_lines

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_examples(self, data_path, data_type, dataset_name):
        if dataset_name.lower() == 'ditk':
            lines = [line.strip() for line in open(data_path)]
            data = self._read_data_from_list(self.ditk_ner_to_conll2003(lines))
        elif dataset_name.lower() == 'conll2003':
            data = self._read_data(data_path)

        return self._create_example(
            data, data_type
        )

    def get_labels(self):
        return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.model_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def write_labels(labels, mode):
    if mode == "test":
        path = os.path.join(FLAGS.model_dir, "label_" + mode + ".txt")
        with open(path, 'a') as f:
            labels = '[CLS]\n' + "\n".join(label for label in labels) + '\n[SEP]\n'
            f.write(labels)


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    write_tokens(ntokens, mode)
    write_labels(labels, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 13])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, 13, [1, 2, 4, 5, 6, 7, 8, 9], average="macro")
                recall = tf_metrics.recall(label_ids, predictions, 13, [1, 2, 4, 5, 6, 7, 8, 9], average="macro")
                f = tf_metrics.f1(label_ids, predictions, 13, [1, 2, 4, 5, 6, 7, 8, 9], average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


class MultiNer(Ner):

    def __init__(self):
        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        if FLAGS.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length.max_seq_length, self.bert_config.max_position_embeddings))

        self.processor = NerProcessor()
        self.label_list = self.processor.get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

        self.tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name,
                zone=FLAGS.tpu_zone,
                project=FLAGS.gcp_project
            )

        self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=self.tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=self.is_per_host)
        )

        self.train_examples = None
        self.num_train_steps = None
        self.num_warmup_steps = None

    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Args:
            data: data in proper [arbitrary] format for train or test. [i.e. format of output from read_dataset]
        Returns:
            ground_truth: [tuple,...], i.e. list of tuples. [SAME format as output of predict()]
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        Raises:
            None
        """
        pass

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Args:
            file_dict: dictionary
                 {
                    "train": dict, {key="file description":value="file location"},
                    "dev" : dict, {key="file description":value="file location"},
                    "test" : dict, {key="file description":value="file location"},
                 }
            dataset_name: str
                Name of the dataset required for calling appropriate utils, converters
        Returns:
            data: data in arbitrary format for train or test.
        Raises:
            None
        """
        train_examples = self.processor.get_examples(file_dict["train"]["data"], "train", dataset_name)
        test_examples = self.processor.get_examples(file_dict["test"]["data"], "test", dataset_name)
        dev_examples = self.processor.get_examples(file_dict["dev"]["data"], "dev", dataset_name)

        self.num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * FLAGS.warmup_proportion)

        return train_examples, test_examples, dev_examples

    def train(self, train_data, *args, **kwargs):
        """
            Trains a model on the given training data

            Args:
                train_data: list

            Returns:
                (Optional) : trained model in applicable formats.
                None: if the model is stored internally.
        """
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size
        )

        train_file = os.path.join(FLAGS.model_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_data, self.label_list, FLAGS.max_seq_length, self.tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_data))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def predict(self, test_data, *args, **kwargs):
        """
        Predicts on the given input data. Assumes model has been trained with train()
        Args:
            data: iterable of arbitrary format. represents the data instances and features you use to make predictions
                Note that prediction requires trained model. Precondition that class instance already stores trained model
                information.
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size
        )

        token_path = os.path.join(FLAGS.model_dir, "token_test.txt")
        with open('./output/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        if os.path.exists(os.path.join(FLAGS.model_dir, "label_test.txt")):
            os.remove(os.path.join(FLAGS.model_dir, "label_test.txt"))
        # predict_examples = self.processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.model_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(test_data, self.label_list,
                                                 FLAGS.max_seq_length, self.tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(test_data))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.model_dir, "label_prediction.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = "\n".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)

        res = []
        tokens = [line.strip('\n') for line in open(os.path.join(FLAGS.model_dir, "token_test.txt"))]
        prediction = [line.strip('\n') for line in open(os.path.join(FLAGS.model_dir, "label_prediction.txt"))]
        labels = [line .strip('\n') for line in open(os.path.join(FLAGS.model_dir, "label_test.txt"))]

        with open(os.path.join(FLAGS.model_dir, "prediction.txt"), 'w') as f:
            for i in range(len(prediction)):
                f.write('{} {} {}\n'.format(tokens[i], prediction[i], labels[i]))
                res.append([(None, len(tokens[i]), tokens[i], prediction[i])])

        return res

    # - start index: int, the index of the first character of the mention span. None if not applicable.
    # - span: int, the length of the mention. None if not applicable.
    # - mention text: str, the actual text that was identified as a named entity. Required.
    # - mention type: str, the entity/mention type. None if not applicable.

    def evaluate(self, test_data, predictions=None, groundTruths=None, *args, **kwargs):
        """
        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]
        Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.
        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
        Raises:
            None
        """
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size
        )

        eval_file = os.path.join(FLAGS.model_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            test_data, self.label_list, FLAGS.max_seq_length, self.tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(test_data))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(test_data) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.model_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        f1 = float(result['eval_f'])
        precision = float(result['eval_precision'])
        recall = float(result['eval_recall'])

        return f1, precision, recall

    def save_model(self, model_path):
        """
        :param model_path: Where to save the model - Optional function
        :return:
        """
        pass

    def load_model(self, model_path):
        """
        :param model_path: From where to load the model - Optional function
        :return:
        """
        pass


def main(input_file_path):

    tf_model = MultiNer()

    file_dict = {
        "train": {
            "data": "./data/train.txt"
        },
        "dev": {
            "data": "./data/dev.txt"
        },
        "test": {
            "data": input_file_path
        }
    }

    train_data, test_data, dev_data = tf_model.read_dataset(file_dict, 'ditk')
    tf_model.train(train_data)

    predictions = my_model.predict(test_data)
    # print("prediction file path : {}".format(os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))))

    tf_model.evaluate(test_data)

    return os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))


if __name__ == "__main__":
    input_file_path = "/data/test.txt"
    output_file_path = main(input_file_path)
    print(output_file_path)
