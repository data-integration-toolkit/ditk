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
		return self.sem_sim.getSimilarity(e1, e2)

	def load_gene_annotation(self, file_path="./dataset/example_gene_association.fb"):
		print('Loading annotation file: ', file_path)
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
	def load_def_file(self, file_path="./dataset/example_categories.txt"):
		print('Loading category file: ', file_path)
		self.cat_def = loadGOcategoryDefinitionFile(file_path)

	def run(self, input_file):
		method = self.option[OPT_METHOD]
		threshold = self.option[OPT_METHOD_THRESHOLD]
		cpu = 1
		sim_index = 'go_sim.txt'

		cat_def = self.cat_def

		org_goid_dict = self.org_goid_dict

		print('Loading genes: ', input_file)
		gene_list = loadGeneList(input_file)

		print('-----------------')
		print('Categorizing... ')
		ct = sorted(cat_def.keys())
		all_categories = ct + [ CATEGORY_NO_ANNOTATION ]


		gene_category = None

		gene_category = process(cat_def, gene_list, org_goid_dict, self.option)

		#out = inpu_file + '.result.txt'

		return report(all_categories, gene_category)

def main(input_path):
	cate = Categorizer()
	cate.load_gene_annotation()
	cate.load_def_file()
	gene_category = cate.run(input_path)
	output_path = "./output.txt"
	f = open("output.txt", "w")
	res = ''	
	for g in gene_category:
		res += g + ":"
		res += ','.join([c + "(" + str(gene_category[g][c]) + ")" for c in gene_category[g]])
		res += '\n'
	f.write(res)
	return output_path
	
if __name__ == "__main__":
	main("./dataset/example_genes.txt")
