import abc

class ProstBase(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def load_rdf(self, input_rdf, output_dir, path_to_jar):
        """This function will load an rdf file into the PROST format (VP and Property Table) into the HDFS 
            Args:
                param1 (str): Input RDF File.
                param2 (str): output directory
                param2 (str): Path to Jar file of PROST	Loader			

            Returns:
                nothing - Files will be saved in output directory on HDFS
        """
        pass

    @classmethod
    @abc.abstractmethod
    def query_executor(sparql_query_file_or_dir, db_dir, output_dir, path_to_jar):
        """This function will execute a Spark SQL query

        Args:
            param1 (str): SPARQL query file
            param1 (str): database directory. 
            param3 (str): output directory           
  			param2 (str): Spark cluster
            param2 (str): Path to Jar file of PROST	Query Executor			

            Returns:
                nothing - Files will be saved in the output directory.
        """
        pass

