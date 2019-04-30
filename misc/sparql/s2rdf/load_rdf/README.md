# Load RDF using [S2RDF](https://github.com/aschaetzle/S2RDF)

This code will launch four Spark jobs on Google Cloud (Dataproc) which will load a small RDF file into Google storage as Apache Parquet tables.

The first job will load the Vertical Parition (VP) tables while the remaing three jobs will load the Extended Vertical Parition (ExtVP) tables.

Upon completion, this code will download two csv files.  The first will contain statistics about the VP tables (`s2rdf_vp_stats.csv`) and the second (`s2rdf_extVp_stats.csv`) will contain statistics about the ExtVP table.  For other data sets there could be up to three statistics files for the ExtVP tables