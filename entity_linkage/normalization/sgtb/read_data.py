import numpy as np
import json

def make_idx_data(docs, ncand=30, skip=False):
    """
    Convert data to fit sklearn regression trees.

    Inputs
    -------
        docs: a document list '[[[(mention_str, offset, wikiID), [(entity, label), [feature]]]]]'
        ncand: number of entity candidates for a mention
        skip: whether to skip mentions whose gold entities are not in candidates
              (used for training data only)
        
    Outputs
    -------
        X: a local feature matrix, label and mention indices arraries
        y: a label array
        indices: a list of pair '(a list of mention indices, a list of gold entity ids)' 
        ent_ids: wikiID for entities (used for querying entity-entity features)
    """
    X, y, indices, ent_ids = [], [], [], []
    i = 0
    for doc in docs:
        doc_idx = []
        gold_ids, skip_ids = [], [] 
        for mentcand in doc:
            ment_idx = []
            flag = False
            tX, ty, tids = [], [], []
            for entcand in mentcand[1][:ncand]:
                tX.append(entcand[1])
                ty.append(entcand[0][1])
                if ty[-1] == 1: flag = True
                tids.append(entcand[0][0])
                ment_idx.append(i)
                i += 1
            if skip and not flag:
                i = len(y)
                continue
            else:
                X += tX
                y += ty
                ent_ids += tids
            if len(ment_idx) > 0: 
                doc_idx.append(ment_idx)
                gold_ids.append(mentcand[0][-1])
            else: # must be a false negative
                skip_ids.append(mentcand[0][-1]) 
        if len(doc_idx) > 0: 
            # append skip_ids after gold_ids, in order to properly evaluate
            # note len(doc_idx) != len(gold_ids+skip_ids)
            indices.append((doc_idx, gold_ids+skip_ids))
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')
    return X, y, indices, ent_ids

def read_data(dataset, split_ratio, params):

    print("Loading data...")

    # data
    processed_docs = []
    with open(dataset, 'rb') as f:
        for line in f:
            processed_docs.append(json.loads(line.strip()))

    train_dev_split_idx = int(len(processed_docs) * split_ratio[0])
    dev_test_split_idx = int(len(processed_docs) * (split_ratio[0] + split_ratio[1]))

    train_docs, dev_docs, test_docs = processed_docs[ : train_dev_split_idx],\
                processed_docs[train_dev_split_idx : dev_test_split_idx],\
                processed_docs[dev_test_split_idx : ]

    train_set = make_idx_data(train_docs, params.get('num_candidate', 30), skip=True)
    dev_set = make_idx_data(dev_docs, params.get('num_candidate', 30))
    test_set = make_idx_data(test_docs, params.get('num_candidate', 30))

    print("Loading data... finished!")

    return train_set, dev_set, test_set
