import abc

class GraphSimilarity(abc.ABC):
	
	@abc.abstractmethod
	def __init__(self, go_file_path, gene_annotation_fiile_path):
		pass

	@abc.abstractmethod
	def load_gene_ontology(self, file_path):
		"""
		function used for load gene ontology file
		gene ontology should in OBO format
		described in http://owlcollab.github.io/oboformat/doc/obo-syntax.html
		"""
		pass

	@abc.abstractmethod
	def load_gene_annotation(self, file_path):
		"""
		function used for load gene annotation file
		gene annotation file should in GAF format
		described in http://geneontology.org/docs/go-annotation-file-gaf-format-2.1/
		"""
		pass

	@abc.abstractmethod
	def similarity(self, e1, e2):
		"""
		function returned the similarity score between two entities if those two entities can be found
		"""
		pass

	@abc.abstractmethod
	def pre_compute(self, e_list):
		"""
		function used for computing all similarity score between a list of entities piars and store them
		"""
		pass

	@abc.abstractmethod
	def evaluate(self, dataset):
		"""
		function used for evaluation with PPI dataset
		"""
		pass
