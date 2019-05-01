from tcss import *

class TSCC:
    def __init__(self):
        #self.dmain_cutoff = domain_cutoff 
        self.g = GOGraph()
        self.run = {'C':self.g._cellular_component, 'P':self.g._biological_process, 'F':self.g._molecular_function}
        self.ont = {'C':"Cellular Component", 'P':"Biological Process", 'F':"Molecular Function"}
        self.code = "TEA"

    def load_gene_ontology(self, file_path="./data/gene_ontology.obo.txt"):
        print "loading ontology data............."
        self.g._obo_parser(file_path)

    def load_gene_annotation(self, file_path='./data/gene_association.sgd'):
        print "loading gene data............."
        self.g._go_annotations(file_path, self.code)

    def cluster(self, ontology = ["C:2.4"]):
        objs = {}
        for i in ontology:
            i = i.split(":")
            print "working with %s ontology....." % self.ont[i[0]]
            objs[i[0]] = self.run[i[0]]()
            objs[i[0]]._species()
            objs[i[0]]._clustering(float(i[1]))
            self.objs = objs
    def similarity(self, e1, e2):
        result = calculate_semantic_similarity(self.objs, e1, e2, False)
        return result
    def similarity_for_file(self, file_path):
        res = []
        for i in open(file_path):
            l = i.rstrip().split(",")
            e1, e2 = l[0], l[1]
            res.append(self.similarity(e1, e2))
        return res


def main(input_path):
    tscc = TSCC()
    tscc.load_gene_ontology()
    tscc.load_gene_annotation()
    tscc.cluster()
    output_path = tscc.similarity_for_file("data/positives.sgd.c")
    print(output_path)
    return output_path


if __name__ == "__main__":
    main("data") 
