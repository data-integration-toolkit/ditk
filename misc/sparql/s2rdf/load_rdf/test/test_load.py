#!/usr/bin/env python3

import unittest
import os
import sys

class TestS2RDFLoad(unittest.TestCase):

    def setUp(self):
        bucket = 'dataproc-11983e44-dcde-4c2a-841f-52a8286e57f5-us-west1'
        region = 'us-west1'
        cluster = 'cluster-4a25'
        project = 'csci-548-project-sp2019-237020'

        module_dir = os.path.dirname(__file__)
        sys.path.append(os.path.join(module_dir, '../..'))
        from s2rdf import S2RDF

        # S2RDF example
        # s = S2RDF(bucket, region, cluster, project)
        # s2rdf_jar_path = "csci548-jars/s2rdf/"
        # s.load_rdf("small/dataset.nt",s2rdf_jar_path + 'dataset-creator-all.jar')
    
    def test_load_rdf(self):
        self.assertTrue(os.path.exists('./s2rdf_vp_stats.csv'))
        self.assertTrue(os.path.exists('./s2rdf_extVp_stats_1.csv'))

if __name__ == '__main__':
    unittest.main()
    