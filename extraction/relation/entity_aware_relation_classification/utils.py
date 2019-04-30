import tensorflow as tf
import numpy as np

class2label = {
    'semeval2010': {'Other': 0,
                    'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                    'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                    'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                    'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                    'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                    'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                    'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                    'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                    'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18},
    'ddi2013': {'false': 0,
                'advise': 1, 'effect': 2,
                'int': 3, 'mechanism': 4},
    'nyt': {'None': 0,
            '/business/company/advisors': 1, '/business/company/founders': 2,
            '/business/company/industry': 3, '/business/company/major_shareholders': 4,
            '/business/company/place_founded': 5, '/business/company_shareholder/major_shareholder_of': 6,
            '/business/person/company': 7, '/location/administrative_division/country': 8,
            '/location/country/administrative_divisions': 9, '/location/country/capital': 10,
            '/location/location/contains': 11, '/location/neighborhood/neighborhood_of': 12,
            '/people/deceased_person/place_of_death': 13, '/people/ethnicity/geographic_distribution': 14,
            '/people/ethnicity/people': 15, '/people/person/children': 16,
            '/people/person/ethnicity': 17, '/people/person/nationality': 18,
            '/people/person/place_lived': 19, '/people/person/place_of_birth': 20,
            '/people/person/profession': 21, '/people/person/religion': 22,
            '/sports/sports_team/location': 23, '/sports/sports_team_location/teams': 24}
}

label2class = {
    'semeval2010': {0: 'Other',
                    1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                    3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                    5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                    7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                    9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                    11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                    13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                    15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                    17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'},
    'ddi2013': {0: 'false',
                1: 'advise', 2: 'effect',
                3: 'int', 4: 'mechanism'},
    'nyt': {0: 'None',
            1: '/business/company/advisors', 2: '/business/company/founders',
            3: '/business/company/industry', 4: '/business/company/major_shareholders',
            5: '/business/company/place_founded', 6: '/business/company_shareholder/major_shareholder_of',
            7: '/business/person/company', 8: '/location/administrative_division/country',
            9: '/location/country/administrative_divisions', 10: '/location/country/capital',
            11: '/location/location/contains', 12: '/location/neighborhood/neighborhood_of',
            13: '/people/deceased_person/place_of_death', 14: '/people/ethnicity/geographic_distribution',
            15: '/people/ethnicity/people', 16: '/people/person/children',
            17: '/people/person/ethnicity', 18: '/people/person/nationality',
            19: '/people/person/place_lived', 20: '/people/person/place_of_birth',
            21: '/people/person/profession', 22: '/people/person/religion',
            23: '/sports/sports_team/location', 24: '/sports/sports_team_location/teams'}
}


def initializer():
    return tf.keras.initializers.glorot_normal()


def load_word2vec(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(word2vec_path))
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return initW


def load_glove(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW
