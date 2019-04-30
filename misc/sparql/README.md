# SPARQL querying and loading rdf graphs using Apache Spark
More researchers are looking into how distributed systems such as Hadoop can be used to efficiently load and query RDF data sets.  The technology that has shown the most promise is [Apache Spark](https://spark.apache.org/).  The software in this repo both use the Hdoop File System as the RDF store and use Spark to quickily query the RDF dataset.

# [S2RDF (SPARQL on Spark for RDF)](https://github.com/aschaetzle/S2RDF)
**[Alexander Schatzle, Martin Przyjaciel-Zablocki, Simon Skilevic, Georg Lausen. S2RDF: RDF Querying with SPARQL on Spark. Proceedings of the VLDB Endowment, Volume 9, No. 10, 2016](http://www.vldb.org/pvldb/vol9/p804-schaetzle.pdf)**

S2RDF defines a new partitioning schema for RDF called Extended Vertical Partitioning (ExtVP).  ExtVP is derived from existing Vertical Partitioning (VP) tables. VP tables are two column tables for each predicate (where the columns are the subject and object respectively). ExtVP tables are the results of performing left joins on two VP tables based on three correlation triple patterns (subject-subject, subject-object and object-subject).  This means each predicate can have up to 6 tables (1 VP, 5 Ext VP).  A selectivity factor is the ratio of the size of the new ExtVP table vs. the size of the VP table that was used to generate.  This can be used to decrease the amout of ExtVP tables that are generated.  For query translation, the table with the lowest selectivity factor for each basic graph pattern in the triple is chosen. Each SELECT statement is then joined to create the (Spark) SQL query.

- Input: RDF file in N-Triples format/ SPARQL query file
- Output: CSV file/ [Apache Parquet](https://parquet.apache.org/) file
- Dataset: [Waterloo Diversity Test Suite](https://dsg.uwaterloo.ca/watdiv/) 100k/1 million


# [PRoST (Partitioned RDF on Spark Tables)](https://github.com/tf-dbis-uni-freiburg/PRoST#prost-partitioned-rdf-on-spark-tables)
**[Matteo Cossu, Michael Färber, Georg Lausen. PRoST: Distributed Execution of SPARQL Queries Using Mixed Partition Strategies. 21st International Conference on Extending Database Technology, 2018](https://github.com/tf-dbis-uni-freiburg/PRoST)**

PRoST stores the data twice using Vertical Partition Tables and Property Tables.  A property table consists of rows for each distinct subject (the key) and the columns contain all the object values for that subject.  The columns are identified by the property (predicate) to which they belong.  Property tables are great for subject-subject triple patterns.  PRoST models SPARQL queries using Join Trees.  A Join Tree is such that nodes are predicates.  Basic graph patterns that are subject-subject are represented together as a single node and marked with a special label denoting that a property table should be used.  All other nodes will use VP tables.  Join Trees are joined bottom-up.  Node placement is based on priority determined by the total number of triples and the number of distinct subjects for each predicate.  Triples with literals get high priority,

- Input: RDF file in N-Triples format/ SPARQL query file
- Output: CSV file/ [Apache Parquet](https://parquet.apache.org/) file
- Dataset: [Waterloo Diversity Test Suite](https://dsg.uwaterloo.ca/watdiv/) 100k/1 million
