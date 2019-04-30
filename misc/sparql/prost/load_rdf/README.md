# Load RDF using [PRoST](https://github.com/tf-dbis-uni-freiburg/PRoST)

This code will launch a Spark job on Google Cloud (Dataproc) which will load a small RDF file into Google storage as Apache Parquet tables.

The job will load the Vertical Parition (VP) tables and the property table.

Upon completion, this code will download a csv file, `prost_vp_stats.csv`.  It will contain statistics about the VP tables that were loaded.