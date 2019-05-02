def clean_text(text):
    text = text.replace('\n', ' ')
    return text


def get_entities_from_tuple(words, embeddings, ner, trans_prob):
    
    import gcn_ner.utils as utils

    sentence = utils.aux.create_full_sentence(words)
    A_fw, A_bw, tags, X = utils.aux.create_graph_from_sentence_and_word_vectors(sentence, embeddings)
    predictions = ner.predict_with_viterbi(A_fw, A_bw, X, tags, trans_prob)
    entities = [utils.aux.get_entity_name(p) for p in predictions]
    return entities


def erase_non_entities(all_words, all_entities, all_idx, all_span):
    return [(w, e, i, s) for w, e, i, s in zip(all_words, all_entities, all_idx, all_span)]


def join_consecutive_tuples(tuples):
    for i in range(len(tuples) - 1):
        curr_type = tuples[i][1]
        curr_end_idx = tuples[i][2][1]
        next_type = tuples[i + 1][1]
        next_start_idx = tuples[i + 1][2][0]
        if curr_type == next_type and curr_end_idx == next_start_idx - 1:
            curr_word = tuples[i][0]
            next_word = tuples[i + 1][0]
            curr_start_idx = tuples[i][2][0]
            next_end_idx = tuples[i + 1][2][1]
            tuples[i + 1] = (curr_word + ' ' + next_word,
                             curr_type,
                             [curr_start_idx, next_end_idx])
            tuples[i] = ()
    tuples = [t for t in tuples if t]
    return tuples

def bio_tagger(tuples):
    flag = 0
    copy_tuples = [0]*len(tuples)
    arr = ['None']*len(tuples)
    for i in range(len(tuples)):
        arr[i] = tuples[i][1]
    for i in range(1, len(tuples)):
        prev = tuples[i-1][1]
        curr = tuples[i][1]
        try:
            if curr == 'O':
                arr[i] = 'O'
            elif (prev == 'O' or prev != curr) and curr != 'O' and (tuples[i+1][1] == 'O' or tuples[i+1][1] != curr):
                arr[i] = 'B-'+curr
            elif prev == curr and flag == 0:
                arr[i-1] = 'B-'+prev
                arr[i] = 'I-'+curr
                flag = 1
            elif prev == curr and flag != 0:
                arr[i] = 'I-'+curr
            elif prev != curr and flag != 0:
                flag = 0
        except:
            continue
    for i in range(len(tuples)):
        copy_tuples[i] = (tuples[i][0], arr[i], tuples[i][2], tuples[i][3])
    return copy_tuples
            

def clean_tuples(all_words, all_entities, all_idx, all_span):
    tuples = erase_non_entities(all_words, all_entities, all_idx, all_span)
    #tuples = join_consecutive_tuples(tuples)
    tuples_bio = bio_tagger(tuples)
    return tuples_bio
