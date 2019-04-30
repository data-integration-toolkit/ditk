# Execute SPARQL query using [PRoST](https://github.com/tf-dbis-uni-freiburg/PRoST)

This code will launch a Spark job on Google Cloud (Dataproc) which a SPARQL query.

The input is the database name in which the VP and Property tables were loaded, the output directory, the SPARQL query and the path to the JAR file that executes the code.

Upon completion, this code will download a csv which contains the results of the query.

The query is located in `small_query.txt`.