# -*- coding: utf-8 -*-

import joblib
import numpy as np
from keras.backend.common import floatx
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.cross_validation import train_test_split

from ntee.model import build_model
from ntee.utils.sentence_detector import OpenNLPSentenceDetector
from ntee.utils.tokenizer import RegexpTokenizer
import click
import os
import re
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from ntee.cli import cli
from ntee.model_reader import ModelReader


def train(db, entity_db, vocab, word2vec, out_file, mode, text_len, dim_size,
          negative, epoch, batch_size, word_static, entity_static, include_title,
          optimizer, dev_size, patience, num_links, random_seed):
    np.random.seed(random_seed)

    click.echo('Initializing weights...')
    word_embedding = np.random.uniform(low=-0.05, high=0.05,
                                       size=(vocab.word_size, dim_size))
    word_embedding = np.vstack([np.zeros(dim_size), word_embedding])
    word_embedding = word_embedding.astype(floatx())

    entity_embedding = np.random.uniform(low=-0.05, high=0.05,
                                         size=(vocab.entity_size, dim_size))
    entity_embedding = entity_embedding.astype(floatx())

    if word2vec:
        for word in vocab.words():
            try:
                vec = word2vec.get_word_vector(word)
            except KeyError:
                continue

            if vec is not None:
                word_embedding[vocab.get_word_index(word) + 1] = vec

        for entity in vocab.entities():
            try:
                vec = word2vec.get_entity_vector(entity)
            except KeyError:
                continue
            if vec is not None:
                entity_embedding[vocab.get_entity_index(entity)] = vec / np.linalg.norm(vec, 2)

    tokenizer = RegexpTokenizer()

    if mode == 'sentence':
        sentence_detector = OpenNLPSentenceDetector()

    def generate_data(keys, count_links=False, shuffle=True, loop=True):
        num_entities = entity_embedding.shape[0]
        labels = np.zeros((batch_size, negative + 1), dtype=np.int)
        labels[:, 0] = 1

        while True:
            word_batch = []
            entity_batch = []

            if shuffle:
                keys = np.random.permutation(keys)

            for key in keys:
                value = db[key]
                text = value['text']
                links = value['links']

                target_data = []
                if mode == 'paragraph':
                    target_data = [(text, links)]
                    if include_title:
                        target_data[0][1].append((None, key.decode('utf-8'), None))

                elif mode == 'sentence':
                    for (start, end) in sentence_detector.sent_pos_detect(text):
                        target_data.append((
                            text[start:end],
                            [(l[0], l[1], (l[2][0] - start, l[2][1] - start))
                             for l in links if start <= l[2][0] < end]
                        ))
                        if include_title:
                            target_data[-1][1].append((None, key.decode('utf-8'), None))
                else:
                    raise NotImplementedError()

                for (target_text, target_links) in target_data:
                    word_indices = []
                    word_offsets = []
                    for token in tokenizer.tokenize(target_text):
                        word_index = vocab.get_word_index(token.text.lower())
                        if word_index is not None:
                            word_indices.append(word_index + 1)
                            word_offsets.append(token.span[0])

                    positive_ids = [
                        vocab.get_entity_index(entity_db.resolve_redirect(title))
                        for (_, title, _) in target_links
                    ]
                    positive_id_set = frozenset([o for o in positive_ids if o is not None])

                    for (positive_id, (_, title, span)) in zip(positive_ids, target_links):
                        if positive_id is None:
                            continue

                        if not word_indices:
                            continue

                        if count_links:
                            yield 1
                            continue

                        negatives = []
                        while True:
                            negative_id = np.random.randint(0, num_entities)
                            if negative_id not in positive_id_set:
                                negatives.append(negative_id)
                            if len(negatives) == negative:
                                break

                        word_batch.append(word_indices)

                        entity_indices = [positive_id] + negatives
                        entity_batch.append(entity_indices)

                        if len(word_batch) == batch_size:
                            yield ([pad_sequences(word_batch, maxlen=text_len),
                                    np.vstack(entity_batch)], labels)

                            word_batch = []
                            entity_batch = []

            if word_batch:
                yield ([pad_sequences(word_batch, maxlen=text_len),
                        np.vstack(entity_batch)], labels[:len(word_batch)])

            if not loop or count_links:
                break

    (train_keys, dev_keys) = train_test_split(db.keys(), test_size=dev_size,
                                              random_state=random_seed)

    if num_links is None:
        click.echo('Counting links...')
        with click.progressbar(train_keys) as bar:
            num_links = sum(list(generate_data(bar, count_links=True, shuffle=False)))

        click.echo('The number of links: %d' % num_links)

    dev_data = list(generate_data(dev_keys, loop=False))
    dev_data = (
        [np.vstack([d[0][0] for d in dev_data]),
         np.vstack([d[0][1] for d in dev_data])],
        np.vstack([d[1] for d in dev_data])
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(out_file + '_checkpoint.h5',
                                     monitor='val_acc', save_best_only=True))
    if patience:
        callbacks.append(EarlyStopping(monitor='val_acc', patience=patience))

    model = build_model(
        text_len=text_len,
        negative_size=negative,
        optimizer=optimizer,
        word_size=word_embedding.shape[0],
        entity_size=entity_embedding.shape[0],
        dim_size=word_embedding.shape[1],
        word_static=word_static,
        entity_static=entity_static,
        word_embedding=word_embedding,
        entity_embedding=entity_embedding,
    )

    model.fit_generator(generate_data(train_keys),
                        samples_per_epoch=num_links,
                        nb_epoch=epoch,
                        validation_data=dev_data,
                        max_q_size=1000,
                        callbacks=callbacks)

    db.close()

    word_embedding = model.get_layer('word_embedding').get_weights()[0][1:]
    entity_embedding = model.get_layer('entity_embedding').get_weights()[0]

    ret = dict(
        word_embedding=word_embedding,
        entity_embedding=entity_embedding,
        vocab=vocab,
    )
    ret['W'] = model.get_layer('text_layer').get_weights()[0]
    ret['b'] = model.get_layer('text_layer').get_weights()[1]

    joblib.dump(ret, out_file + '.joblib', protocol=-1)

@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('dataset_file', type=click.File())
def evaluate_sick(model_file, dataset_file):
    reader = ModelReader(model_file)

    predicted = []
    correct = []
    x,y = [],[]
    f = open('sick_pear.txt', 'w')
    for (n, line) in enumerate(dataset_file):
        if n == 0:
            continue

        data = line.rstrip().decode('utf-8').split('\t')
        sent1 = data[1]
        sent2 = data[2]
        score = float(data[4])
        fold = data[11]
        if fold == 'TRIAL':
            continue

        correct.append(float(score))

        vec1 = reader.get_text_vector(sent1)
        vec2 = reader.get_text_vector(sent2)
        predicted.append(1.0 - cosine(vec1, vec2))

        res = str(pearsonr(correct, predicted)) + '\n'

        x.append(pearsonr(correct, predicted)[0])
        y.append(pearsonr(correct, predicted)[1])

        f.write(res)
    plt.plot(x, y)

    plt.savefig('sick_pear.png', dpi=100)
    f.close()

    mean_sq_err = mean_squared_error(correct, predicted)
    click.echo('%.4f (pearson) %.4f (spearman) %.4f (Mean Squared Error)' % (
        pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0], mean_sq_err
    ))


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('dataset_dir', type=click.Path(exists=True))
def evaluate_sts(model_file, dataset_dir):
    reader = ModelReader(model_file)

    for file_name in sorted(os.listdir(dataset_dir)):
        match_obj = re.match(r'^STS\.input\.(.*)\.txt', file_name)
        if not match_obj:
            continue

        name = match_obj.group(1)

        predicted = []
        correct = []
        with open(os.path.join(dataset_dir, file_name)) as input_file:
            with open(os.path.join(dataset_dir, 'STS.gs.' + name + '.txt')) as gs_file:
                for (line, score) in zip(input_file, gs_file):
                    score = score.rstrip()
                    if not score:
                        continue
                    score = float(score)

                    (sent1, sent2) = line.rstrip().decode('utf-8').split('\t')
                    correct.append(score)

                    vec1 = reader.get_text_vector(sent1)

                    vec2 = reader.get_text_vector(sent2)

                    predicted.append(1.0 - cosine(vec1, vec2))

        mean_sq_err = mean_squared_error(correct, predicted)
        click.echo('%s: %.4f (pearson) %.4f (spearman) %.4f (Mean Squared Error)' % (
            name, pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0], mean_sq_err
        ))


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('dataset_file', type=click.File())
def evaluate_semeval(model_file, dataset_file):
    reader = ModelReader(model_file)

    predicted = []
    correct = []
    x, y = [], []
    f = open('pear.txt', 'w')
    for (n, line) in enumerate(dataset_file):
        if n == 0:
            continue

        data = line.rstrip().decode('utf-8').split('\t')
        sent1 = data[5]
        sent2 = data[6]
        score = float(data[4])

        correct.append(float(score))

        vec1 = reader.get_text_vector(sent1)
        vec2 = reader.get_text_vector(sent2)
        predicted.append(1.0 - cosine(vec1, vec2))

        # vectorizer = TfidfVectorizer()
        # response = vectorizer.fit_transform([sent1 , sent2])


        res = str(pearsonr(correct,predicted)) + "\n"
        x.append(pearsonr(correct,predicted)[0])
        y.append(pearsonr(correct,predicted)[1])


        f.writelines(res)
    plt.plot(x,y)

    plt.savefig('semeval_pear.png', dpi=100)


    f.close()
    mean_sq_err = mean_squared_error(correct, predicted)


    click.echo('%.4f (pearson) %.4f (spearman) %.4f (Mean Squared Error)' % (
        pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0], mean_sq_err
    ))

