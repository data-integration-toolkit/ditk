def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sequence = item[0]
        length = len(sequence)
        try:
            size_to_data_dict[length].append(item)
        except:
            size_to_data_dict[length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets


def train_and_save(dataset, saving_dir, epochs=20, bucket_size=10):
    import random
    import sys
    import pickle
    import numpy as np
    import gcn_ner.utils.aux as aux

    from gcn_ner.ner_model import GCNNerModel
    from ..aux import  create_full_sentence


    sentences = aux.get_all_sentences_train(dataset)
    #sentences = aux.get_all_sentences(dataset)
    print('Computing the transition matrix')
    data, trans_prob = aux.get_data_from_sentences(sentences)
    buckets = bin_data_into_buckets(data, bucket_size)

    gcn_model = GCNNerModel(dropout=0.7)

    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            try:
                gcn_bucket = []
                for item in bucket:
                    #print("a")
                    words = item[0]
                    #print("b")
                    word_embeddings = item[1]
                    #print("c")
                    sentence = create_full_sentence(words)
                    #print("d")
                    tags = item[2]
                    #print("e")
                    label = item[3]
                    #print("f")
                    label = [np.array(l, dtype=np.float32) for l in label]
                    #print("g")
                    A_fw, A_bw, _, X = aux.create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                    #print("h")
                    gcn_bucket.append((A_fw, A_bw, X, tags, label))
                    #print("i")
                if len(gcn_bucket) > 1:
                    #print("j")
                    gcn_model.train(gcn_bucket, trans_prob, 1)
                    #print("k")
            except:
               #print("l")
               pass
        save_filename = saving_dir + 'ner-gcn-' + str(i) + '.tf'
    return (save_filename, gcn_model)
