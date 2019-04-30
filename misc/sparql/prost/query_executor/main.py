#!/usr/bin/env python3

import os
import sys

def main():
    module_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(module_dir, '..'))
    from prost import Prost

    bucket = 'dataproc-11983e44-dcde-4c2a-841f-52a8286e57f5-us-west1'
    region = 'us-west1'
    cluster = 'cluster-4a25'
    project = 'csci-548-project-sp2019-237020'

    #Prost
    prost_jar_path = 'csci548-jars/prost/'
    p = Prost(bucket, region, cluster, project)
    
    dataset = 'small'
    query = 'small_query.txt'
    p.query_executor(query, dataset, "prost-workspace/{}.{}.results".format(dataset,query[0:-4]),prost_jar_path + "query-all.jar")

main()