This model assumes we are given all the mentions of named entities and a lexicon that maps each mention to a set of entity candidates in a given reference entity database. 

The paper uses raw dataset of AIDA-CoNLL, AQUANT, ACE, but does not provide the preprocessing step to convert txt to json.

The input format(json) is:
(mentioned_entity, offset_pairs, highest_rank_candidate, set of (candidate_entity, label, features)).

The output format is same as submitted in the group.

The evaluation metric is also not in F1. They only provides accuracy and I was not able to extract precision and recall.
