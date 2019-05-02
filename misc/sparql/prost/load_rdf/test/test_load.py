#!/usr/bin/env python3

import unittest
import os
import sys
import csv

class TestProstLoad(unittest.TestCase):

    def setUp(self):
        bucket = 'dataproc-11983e44-dcde-4c2a-841f-52a8286e57f5-us-west1'
        region = 'us-west1'
        cluster = 'cluster-4a25'
        project = 'csci-548-project-sp2019-237020'

        module_dir = os.path.dirname(__file__)
        sys.path.append(os.path.join(module_dir, '../..'))
        from prost import Prost

        bucket = 'dataproc-11983e44-dcde-4c2a-841f-52a8286e57f5-us-west1'
        region = 'us-west1'
        cluster = 'cluster-4a25'
        project = 'csci-548-project-sp2019-237020'

        #Prost
        prost_jar_path = 'csci548-jars/prost/'
        p = Prost(bucket, region, cluster, project)
        p.load_rdf("prost-workspace/small/", "small", prost_jar_path + 'loader-all.jar',)
    
    def test_load_rdf(self):
        self.assertTrue(os.path.exists('./prost_vp_stats.csv'))
        with open('prost_vp_stats.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            num_rows = 0
            for r in reader:
                num_rows = num_rows + 1
            self.assertEquals(5,num_rows)

if __name__ == '__main__':
    unittest.main()
    