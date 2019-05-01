#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:37:45 2019

@author: vivekmishra
"""
import json,sys


def read_data(dataset_name):
    
    
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path:
            ditk_path = path
    
    
    with open(dataset_name) as f:
        tweets = f.read().splitlines()
    
    filename = ditk_path+"/entity_linkage/normalization/lnex/_Data/chennai_50.json"

    # read tweets from file to list
    with open(filename) as f:
        data = json.load(f)

    return tweets,data