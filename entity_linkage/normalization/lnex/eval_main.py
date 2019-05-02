#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:22:11 2019

@author: vivekmishra
"""



from collections import defaultdict
import LNEx as lnex


def do_they_overlap(tub1, tub2):
    '''Checks whether two substrings of the tweet overlaps based on their start
    and end offsets.'''

    if tub2[1] >= tub1[0] and tub1[1] >= tub2[0]:
        return True

def evaluate(anns):
    TPs_count = 0
    FPs_count = 0
    FNs_count = 0
    overlaps_count = 0

    #fns = defaultdict(int)

    count = 0
    one_geolocation = 0
    all_geolocation = 0
    geo_codes_length_dist = defaultdict(int)

    FPs_set = defaultdict(set)
    FNs_set = defaultdict(set)

    for key in list(anns.keys()):

        count += 1

        # skip the development set
        #if dataset != "houston" and count < 500:
        #    continue

        tweet_lns = set()
        lnex_lns = set()
        tweet_text = ""
        
        for ann in anns[key]:
            if ann != "text":
                ln = anns[key][ann]

                tweet_lns.add(((int(ln['start_idx']), int(ln['end_idx'])),
                                    ln['type']))
            else:
                tweet_text = anns[key][ann]

                r = lnex.extract(tweet_text)

                # how many are already disambiguated +++++++++++++++++++++++
                for res in r:
                    if len(res[3]) < 2:
                        one_geolocation += 1

                        #if len(res[3]) == 0:
                            #print res[2]
                    else:
                        geo_codes_length_dist[len(res[3])] += 1

                    all_geolocation += 1
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                lnex_lns = set([x[1] for x in r])

        tweet_lns = set([x[0] for x in tweet_lns if x[1] == "inLoc"])
        
        # True Positives +++++++++++++++++++++++++++++++++++++++++++++++++++
        TPs = tweet_lns.intersection(lnex_lns)

        TPs_count += len(TPs)

        # Left in both sets ++++++++++++++++++++++++++++++++++++++++++++++++
        tweet_lns -= TPs
        lnex_lns -= TPs

        # Find Overlapping LNs to be counted as 1/2 FPs and 1/2 FNs++
        overlaps = set()
        for x in tweet_lns:
            for y in lnex_lns:
                if do_they_overlap(x, y):
                    overlaps.add(x)
                    overlaps.add(y)

        overlaps_count += len(overlaps)

        # remove the overlapping lns from lnex_lns and tweet_lns
        lnex_lns -= overlaps
        tweet_lns -= overlaps

        # False Positives ++++++++++++++++++++++++++++++++++++++++++++++++++
        # lnex_lns = all - (TPs and overlaps and !inLoc)
        FPs = lnex_lns - tweet_lns
        FPs_count += len(FPs)

        if len(FPs) > 0:
            for x in FPs:
                FPs_set[tweet_text[x[0]:x[1]]].add((key,tweet_text[x[0]-2:x[1]+2],x))

        # False Negatives ++++++++++++++++++++++++++++++++++++++++++++++++++
        FNs = tweet_lns - lnex_lns
        FNs_count += len(FNs)

        if len(FNs) > 0:
            for x in FNs:
                FNs_set[tweet_text[x[0]:x[1]]].add((key,tweet_text[x[0]-2:x[1]+2],x))

    '''
    since we add 2 lns one from lnex_lns and one from tweet_lns if they
    overlap the equation of counting those as 1/2 FPs and 1/2 FNs is going
    to be:
        overlaps_count x
            1/2 (since we count twice) x
                1/2 (since we want 1/2 of all the errors made)
    '''

    Precision = TPs_count/(TPs_count + FPs_count + 0.5 * .5 * overlaps_count)
    Recall = TPs_count/(TPs_count + FNs_count + 0.5 * .5 * overlaps_count)
    F_Score = (2 * Precision * Recall)/(Precision + Recall)

    #percentage_disambiguated = one_geolocation/all_geolocation

    return {"precision": Precision, 
            "recall": Recall, 
            "f-score": F_Score}