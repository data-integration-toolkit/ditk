from tcss import *

class TSCC:
    def __init__(self):
        #self.dmain_cutoff = domain_cutoff 
        self.g = GOGraph()
        self.run = {'C':self.g._cellular_component, 'P':self.g._biological_process, 'F':self.g._molecular_function}
        self.ont = {'C':"Cellular Component", 'P':"Biological Process", 'F':"Molecular Function"}
        self.code = "TEA"

    def load_gene_ontology(self, file_path="./data/gene_ontology.obo.txt"):
        print("loading ontology data.............")
        self.g._obo_parser(file_path)

    def load_gene_annotation(self, file_path='./data/gene_association.sgd'):
        print("loading gene data.............")
        self.g._go_annotations(file_path, self.code)

    def cluster(self, ontology = ["C:2.4"]):
        objs = {}
        for i in ontology:
            i = i.split(":")
            print("working with %s ontology....." % self.ont[i[0]])
            objs[i[0]] = self.run[i[0]]()
            objs[i[0]]._species()
            objs[i[0]]._clustering(float(i[1]))
            self.objs = objs
    def similarity(self, e1, e2):
        result = calculate_semantic_similarity(self.objs, e1, e2, False)
        return result
    def pre_compute(self, e_list):
        res = []
        for i in e_list:
            e1, e2 = i[0], i[1]
            res.append(calculate_semantic_similarity(self.objs, e1, e2, False))
        self.res = res
    def similarity_for_file(self, file_path):
        res = []
        for i in open(file_path):
            l = i.rstrip().split(",")
            e1, e2 = l[0], l[1]
            res.append(self.similarity(e1, e2))
        return res
    def evaluate(self, input_file, cutoff, pos_or_neg):
	res = self.similarity_for_file(input_file)
	total = len(res)
        t = 0
        for i in res:
            try:v = float(i.split(":")[1])
            except: continue
            if pos_or_neg:
                if v > cutoff: t+= 1
            else:
                if v <= cutoff: t+= 1
        if pos_or_neg:
            return "true positive is %f" % (1.0 * t / total)
        else:
            return "true negative is %f" % (1.0 * t / total)


def main(input_path):
    tscc = TSCC()
    tscc.load_gene_ontology()
    tscc.load_gene_annotation()
    #tscc.cluster()
    tscc.cluster(ontology = ["P:3.5"])
    #print tscc.evaluate("data/positives.sgd.p",0.5,True)
    output = tscc.similarity_for_file(input_path)
    output_path = "./output.txt"
    f = open(output_path,"w")
    f.write("".join(output))
    return output_path


if __name__ == "__main__":
    main("data/positives.sgd.c") 
