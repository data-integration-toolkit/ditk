from Categorizer import *
import rebuild

class Categorizer:
	def __init__(self, option = None):
		if not option:
			option = {
				OPT_DEF_FILE: None,
				OPT_INPUT_FILE: None,
				OPT_METHOD: None,
				OPT_METHOD_THRESHOLD: None,
				OPT_ANNOTATION_FILE: None,
				OPT_PROGRESS_BAR: None,
				OPT_CPU: 1 }
			option[OPT_METHOD] = OPT_METHOD_MULTIPLE
			option[OPT_METHOD_THRESHOLD] = float(0.3)
		self.option = option
		self.sem_sim = SemanticSimilarity.SEMANTIC_SIMILARITY()

	def similarity(self, e1, e2):
		return self.sem_sim(e1, e2)

	def load_gene_annotation(self, file_path):
		print 'Loading annotation file: ', file_path
		self.option[OPT_ANNOTATION_FILE] = file_path
		org_goid_dict = loadAnnotationFile(self.option[OPT_ANNOTATION_FILE])
		self.org_goid_dict = org_goid_dict
	def load_gene_ontology(self, file_path):
		self.ontology_file = file_path
	def pre_compute(self):
		if not self.option[OPT_ANNOTATION_FILE]:
			print("missing gene annotation file")
		if not self.ontology_file:
			print("missing gene ontology file")
		rebuild.run(self.option[OPT_ANNOTATION_FILE], self.ontology_file)

	def run(self, dataset):
		if dataset == './dataset':
			def_file = dataset + "/example_categories.txt"
			input_file = dataset + "/example_genes.txt"
			annot_file = dataset + "/example_gene_association.fb"
		else:
			def_file = dataset + "/categories.txt"
			input_file = dataset + "/genes.txt"
			annot_file = dataset + "/gene_association.fb"
	
		method = self.option[OPT_METHOD]
		threshold = self.option[OPT_METHOD_THRESHOLD]
		cpu = 1
		sim_index = 'go_sim.txt'

		print 'Loading category file: ', def_file
		cat_def = loadGOcategoryDefinitionFile(def_file)

		print 'Loading annotation file: ', annot_file
		self.load_gene_annotation(annot_file)
		org_goid_dict = self.org_goid_dict

		print 'Loading genes: ', input_file
		gene_list = loadGeneList(input_file)

		print '-----------------'
		print 'Categorizing... '
		ct = cat_def.keys()
		ct.sort()
		all_categories = ct + [ CATEGORY_NO_ANNOTATION ]


		gene_category = None

		gene_category = process(cat_def, gene_list, org_goid_dict, self.option)

		out = input_file + '.result.txt'

		report(all_categories, gene_category, out)

if __name__ == "__main__":
	cate = Categorizer()
	cate.run("./dataset")
	pass
