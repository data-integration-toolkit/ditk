"""
Vectorizing of n-grams.

Run unit tests with ".py"
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse


def get_unique_ngrams(string, n):
    """ Return the set of different n-grams in a string
    """
    # string = '$$' + string + '$$'
    strings = [string[i:] for i in range(n)]
    return set(zip(*strings))


def get_ngrams(string, n):
    """ Return a list with all n-grams in a string.
    """
    # string = '$$' + string + '$$'
    strings = [string[i:] for i in range(n)]
    return list(zip(*strings))


def ngram_similarity1(strings1, strings2, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(strings1) x len(strings2)
    sim(s1, s2) = ||min(x1, x2)||_1 / (||x1||_1 + ||x2||_1 - ||min(x1, x2)||_1)
    """
    unq_strings1 = np.unique(strings1)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    count2 = vectorizer.fit_transform(strings2)
    count1 = vectorizer.transform(unq_strings1)
    sum2 = count2.sum(axis=1)
    SE_dict = {}
    for i, x in enumerate(count1):
        aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
        samegrams = count2.minimum(aux).sum(axis=1)
        allgrams = x.sum() + sum2 - samegrams
        similarity = np.divide(samegrams, allgrams)
        SE_dict[unq_strings1[i]] = np.array(similarity).reshape(-1)
    SE = []
    for s in strings1:
        SE.append(SE_dict[s])
    return np.nan_to_num(np.vstack(SE))


ngram_similarity = ngram_similarity1


def ngram_similarity2(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = 2||min(x1, x2)||_1/ (||x1||_1 + ||x2||_1)
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    count2 = vectorizer.fit_transform(stringsj)
    count1 = vectorizer.transform(unq_stringsi)
    sum_matrix2 = count2.sum(axis=1)
    SE_dict = {}
    for i, x in enumerate(count1):
        aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
        samegrams = count2.minimum(aux).sum(axis=1)
        allgrams = x.sum() + sum_matrix2
        similarity = 2 * np.divide(samegrams, allgrams)
        SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
    SE = []
    for s in stringsi:
        SE.append(SE_dict[s])
    return np.nan_to_num(np.vstack(SE))


def ngram_similarity2_1(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = 2 dot(c1, c2) / (dot(c1, c1) + dot(c2, c2))
    where c is the count vector for n-grams.
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = vectorizer.fit_transform(stringsj).transpose()
    Ci = vectorizer.transform(unq_stringsi)
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                  (1, Cj.shape[1]))
    cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                  (Ci.shape[0], 1))
    similarity = np.divide(2*cij, cii + cjj)
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity2_2(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = 2 dot(p1, p2) / (dot(p1, p1) + dot(p2, p2))
    where p is the presence vector for n-grams.
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = (vectorizer.fit_transform(stringsj) > 0).astype('float64').transpose()
    Ci = (vectorizer.transform(unq_stringsi) > 0).astype('float64')
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                  (1, Cj.shape[1]))
    cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                  (Ci.shape[0], 1))
    similarity = np.divide(2*cij, cii + cjj)
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity3(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(x1, x2) / (dot(x1, x1) + dot(x2, x2) - dot(x1, x2))
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = vectorizer.fit_transform(stringsj).transpose()
    Ci = vectorizer.transform(unq_stringsi)
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                  (1, Cj.shape[1]))
    cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                  (Ci.shape[0], 1))
    similarity = np.divide(cij, cii + cjj - cij)
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity3_2(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(p1, p2) / (dot(p1, p1) + dot(p2, p2) - dot(p1, p2))
    where p is the presence vector for n-grams.
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = (vectorizer.fit_transform(stringsj) > 0).astype('float64').transpose()
    Ci = (vectorizer.transform(unq_stringsi) > 0).astype('float64')
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                  (1, Cj.shape[1]))
    cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                  (Ci.shape[0], 1))
    similarity = np.divide(cij, cii + cjj - cij)
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity4(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(c1, c2) / (dot(c1, c1)^.5 * dot(c2, c2)^.5)
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = vectorizer.fit_transform(stringsj).transpose()
    Ci = vectorizer.transform(unq_stringsi)
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                  (1, Cj.shape[1]))
    cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                  (Ci.shape[0], 1))
    similarity = np.divide(cij, np.multiply(cii, cjj))
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity5(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(p1, p2) / (dot(p1, p1)^.5 * dot(p2, p2)^.5)
    where p is the presence vector
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = (vectorizer.fit_transform(stringsj) > 0).astype('float64').transpose()
    Ci = (vectorizer.transform(unq_stringsi) > 0).astype('float64')
    SE_dict = {}
    cij = Ci.dot(Cj).toarray()
    cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                  (1, Cj.shape[1]))
    cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                  (Ci.shape[0], 1))
    similarity = np.divide(cij, np.multiply(cii, cjj))
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity6(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(c1, c2)
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = vectorizer.fit_transform(stringsj).transpose()
    Ci = vectorizer.transform(unq_stringsi)
    SE_dict = {}
    similarity = Ci.dot(Cj).toarray()
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_similarity7(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    sim(s1, s2) = dot(p1, p2)
    where p is the presence vector
    """
    unq_stringsi = np.unique(stringsi)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = (vectorizer.fit_transform(stringsj) > 0).astype('float64').transpose()
    Ci = (vectorizer.transform(unq_stringsi) > 0).astype('float64')
    SE_dict = {}
    similarity = Ci.dot(Cj).toarray()
    stringsi_dict = {s: i for i, s in enumerate(unq_stringsi)}
    index = [stringsi_dict[s] for s in stringsi]
    similarity = similarity[index]
    return np.nan_to_num(similarity)


def ngram_presence_fisher_kernel(stringsi, stringsj, n):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(stringsi) x len(stringsj)
    kernel fisher with p
    where p is the presence vector
    """
    unq_stringsi = np.unique(stringsi)
    unq_stringsj, count_j = np.unique(stringsj, return_counts=True)
    theta = count_j/sum(count_j)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    Cj = (vectorizer.fit_transform(unq_stringsj) > 0).astype('float64').toarray()
    Ci = (vectorizer.transform(unq_stringsi) > 0).astype('float64').toarray()
    m = Cj.shape[1]
    SE_dict = {}
    for i, p_i in enumerate(Ci):
        gamma = np.zeros(m)
        for j, p_j in enumerate(Cj):
            gamma += (p_j == p_i).astype('float64')*theta[j]
        similarity = []
        for j, p_j in enumerate(Cj):
            sim_j = (p_j == p_i).astype('float64') / gamma
            similarity.append(sim_j.sum())
        SE_dict[unq_stringsi[i]] = np.array(similarity)
    SE = []
    for s in stringsi:
        SE.append(SE_dict[s])
    return np.nan_to_num(np.vstack(SE))

# from Data import *
# data = Data('docs_payments').get_df()
# df = data.df.sample(100000)
# del data
# stringsi = df[df.columns[0]].values.astype(str)
# stringsj = np.unique(stringsi)
# n = 3
# del df
#
# %timeit ngram_similarity1(stringsi, stringsj, n)
# %timeit ngram_similarity2(stringsi, stringsj, n)
# %timeit ngram_similarity3(stringsi, stringsj, n)
# %timeit ngram_similarity4(stringsi, stringsj, n)
# %timeit ngram_similarity5(stringsi, stringsj, n)
#
# A1 = ngram_similarity1(stringsi, stringsj, n)
# A2 = ngram_similarity2(stringsi, stringsj, n)
# A3 = ngram_similarity3(stringsi, stringsj, n)
# A4 = ngram_similarity4(stringsi, stringsj, n)
# A5 = ngram_similarity5(stringsi, stringsj, n)
#
# As = [A1, A2, A3, A4, A5]

# for i, A in enumerate(As):
#     print('A%d max, min: ' % (i+1), A.min(), A.max())
#     print('abs(A1 - A%d): ' % i, np.abs(A1 - A).max())
#     print('')


# def ngram_similarity3(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     sim(s1, s2) = 2 dot(y1, y2) / (dot(y1, y1) + dot(y2, y2))
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     presence_matrix1 = (count1 > 0).astype(int)
#     presence_matrix2 = (count2 > 0).astype(int)
#     sum_matrix2 = presence_matrix2.sum(axis=1)
#     SE_dict = {}
#     for i, x in enumerate(presence_matrix1):
#         aux = presence_matrix2.dot(x.transpose())
#         samegrams = aux.sum(axis=1)
#         allgrams = x.sum() + sum_matrix2
#         similarity = 2 * np.divide(samegrams, allgrams)
#         SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
#     SE = []
#     for s in stringsi:
#         SE.append(SE_dict[s])
#     return np.nan_to_num(np.vstack(SE))
#
#
# def ngram_similarity4(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     2||min(x1, x2)||_1/ (||x1||_1^.5 * ||x2||_1^.5)
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     sum_matrix2 = count2.sum(axis=1)
#     SE_dict = {}
#     for i, x in enumerate(count1):
#         aux = count2.dot(x.transpose())
#         samegrams = aux.sum(axis=1)
#         allgrams = x.sum()**.5 * np.power(sum_matrix2, .5)
#         similarity = np.divide(samegrams, allgrams)
#         SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
#     SE = []
#     for s in stringsi:
#         SE.append(SE_dict[s])
#     return np.nan_to_num(np.vstack(SE))
#
#
# def ngram_similarity5(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     sim(s1, s2) = 2 dot(y1, y2) / (dot(y1, y1)^.5 dot(y2, y2)^.5)
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     presence_matrix1 = (count1 > 0).astype(int)
#     presence_matrix2 = (count2 > 0).astype(int)
#     sum_matrix2 = presence_matrix2.sum(axis=1)
#     SE_dict = {}
#     for i, x in enumerate(presence_matrix1):
#         aux = presence_matrix2.dot(x.transpose())
#         samegrams = aux.sum(axis=1)
#         allgrams = x.sum()**.5 * np.power(sum_matrix2, .5)
#         similarity = np.divide(samegrams, allgrams)
#         SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
#     SE = []
#     for s in stringsi:
#         SE.append(SE_dict[s])
#     return np.nan_to_num(np.vstack(SE))
#
#
# def ngram_similarity6(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     sim(s1, s2) = dot(y1, y2)
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     presence_matrix1 = (count1 > 0).astype(int)
#     presence_matrix2 = (count2 > 0).astype(int)
#     sum_matrix2 = count2.sum(axis=1)
#     aux = presence_matrix1.dot(presence_matrix2.transpose()).toarray()
#     SE_dict = {s: i for i, s in enumerate(unq_stringsi)}
#     SE = []
#     for s in stringsi:
#         SE.append(aux[SE_dict[s]])
#     return np.vstack(SE)
#
#
# def ngram_similarity7(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     sim(s1, s2) = 2 dot(x1, x2) / (dot(x1, x1) + dot(x2, x2))
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     sum_matrix2 = count2.sum(axis=1)
#     SE_dict = {}
#     for i, x in enumerate(count1):
#         aux = count2.dot(x.transpose())
#         samegrams = aux.sum(axis=1)
#         allgrams = x.sum() + sum_matrix2
#         similarity = 2 * np.divide(samegrams, allgrams)
#         SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
#     SE = []
#     for s in stringsi:
#         SE.append(SE_dict[s])
#     return np.nan_to_num(np.vstack(SE))
#
#
# def ngram_similarity8(stringsi, stringsj, n):
#     """ given to arrays of strings, returns the
#     similarity encoding matrix of size
#     len(stringsi) x len(stringsj)
#     sim(s1, s2) = dot(x1, x2) / (dot(x1, x1) + dot(x2, x2) - dot(x1, x2))
#     """
#     unq_stringsi = np.unique(stringsi)
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
#     count2 = vectorizer.fit_transform(stringsj)
#     count1 = vectorizer.transform(unq_stringsi)
#     sum_matrix2 = count2.sum(axis=1)
#     SE_dict = {}
#     for i, x in enumerate(count1):
#         aux = count2.dot(x.transpose())
#         samegrams = aux.sum(axis=1)
#         allgrams = x.sum() + sum_matrix2 - samegrams
#         similarity = np.divide(samegrams, allgrams)
#         SE_dict[unq_stringsi[i]] = np.array(similarity).reshape(-1)
#     SE = []
#     for s in stringsi:
#         SE.append(SE_dict[s])
#     return np.nan_to_num(np.vstack(SE))
#


# n = 3
# from Data import *
# from fns_categorical_encoding import *
# data = Data('medical_charge').get_df()
# stringsi = data.df[[k for k in data.col_action if data.col_action[k] == 'se'
#                     ][0]].astype(str).values[:100000]
# stringsi = np.array([s.lower() for s in stringsi])
# stringsj = np.unique(stringsi)
# stringsi_ = np.array(['$$' + s + '$$' for s in stringsi])
# stringsj_ = np.unique(stringsi_)
#
# %timeit ngram_similarity1(stringsi_, np.unique(stringsj_), 3)
# %timeit ngram_similarity2(stringsi_, np.unique(stringsj_), 3)
# %timeit ngram_similarity3(stringsi_, np.unique(stringsj_), 3)
# %timeit ngram_similarity4(stringsi_, np.unique(stringsj_), 3)
# %timeit ngram_similarity5(stringsi_, np.unique(stringsj_), 3)
# %timeit ngram_similarity6(stringsi_, np.unique(stringsj_), 3)
#
# %timeit categorical_encoding(stringsi, stringsi, 0, '3gram_similarity', '', 1)
#
# A0 = categorical_encoding(stringsi, stringsj, 0, '3gram_similarity', '', 1)
# A1 = ngram_similarity1(stringsi_, np.unique(stringsj_), 3)
# A2 = ngram_similarity2(stringsi_, np.unique(stringsj_), 3)
# A3 = ngram_similarity3(stringsi_, np.unique(stringsj_), 3)
# A4 = ngram_similarity4(stringsi_, np.unique(stringsj_), 3)
# A5 = ngram_similarity5(stringsi_, np.unique(stringsj_), 3)
# A6 = ngram_similarity6(stringsi_, np.unique(stringsj_), 3)
#
# A_ = [A0, A1, A2, A3, A4, A5, A6]
# for A in A_:
#     print(A.max())


def ngrams_count_vectorizer(strings, n):
    """ Return the a disctionary with the count of every
    unique n-gram in the string.
    """
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    count = vectorizer.fit_transform(strings)
    feature_names = vectorizer.get_feature_names()
    return count, feature_names


def ngrams_hashing_vectorizer(strings, n, n_features):
    """ Return the a disctionary with the count of every
    unique n-gram in the string.
    """
    hv = HashingVectorizer(analyzer='char', ngram_range=(n, n),
                           n_features=n_features, norm=None,
                           alternate_sign=False)
    hash_matrix = hv.fit_transform(strings)
    return hash_matrix
