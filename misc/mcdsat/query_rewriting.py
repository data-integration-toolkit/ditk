import abc

class Query_Rewriting(abc.ABC):
	"""
	Parent class which implements various methods for Query Rewriting and MCDs Genreration
	"""	
	def __init__(self):
		"""
		Task -- Initialiazer function for Query Rewriting. Defines all the class members which represent the following :
		
		self.query_file -- string -- path to the query file
		self.views_file -- sring -- path to the views file
		self.algorithm -- string -- algorithm being used for query rewriting, ec. mcdsat or minicon
		self.rw_flag -- boolean -- given task for this object is query rewriting
		self.mcd_flag -- boolean -- given task for this object is mcds generation
		self.query_mcds -- string -- path to the file containing all the generated query mcds using one of the query rewriting algorithm
		self.query_rewritings -- string -- path to the file containing all the possible query rewritings using one the query rewriting algorithms
		"""
		self.query_file = ""
		self.views_file = ""
		self.algorithm = ""
		self.rw_flag = bool
		self.mcd_flag = bool 
		self.query_rewritings = ""
		self.query_mcds = ""
	
	@classmethod
	@abc.abstractmethod
	def read_input(self, *args, **kwargs):
		"""
		Task -- Reads the query file and the views file and stores them in the class data members

		Input:
		query_file -- string -- path to the query file
		views_file -- string -- path to the views file

		Result:
		The class data members are populated appropriately
		"""
		pass

	@classmethod
	@abc.abstractmethod
	def generate_query_rewritings(self, *args, **kwargs):
		"""
		Task -- Generate Query Rewritings for the input query using the provided views file

		Input:
		query_file -- string -- path to the query file
		views_file -- string -- path to the views file

		Return:
		rewriting_file -- string -- path to the file containing all the possible query rewritings for the given query and views
		"""
		pass

	@classmethod
	@abc.abstractmethod
	def generate_MCDs(self, *args, **kwargs):
		"""
		Task -- Generate MCDs based on the query and the views using a specific query rewriting algorithm

		Input:
		query_file -- string -- path to the query file
		views_file -- string -- path to the views file

		Return:
		mcds_file -- string -- path to the file containing all the generated MCDs for the given query and views
		"""
		pass

