'''
Created on 2010-07-19

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

from ontology import GOGraph

def load_semantic_similarity(ontology_file, gene_file, ontology, code):
    '''
    Calls functions for loading and processing data.
    '''
    objs = {}
    ontology = ontology.split(",")
    g = GOGraph()
    print "loading data............."
    g._obo_parser(ontology_file)
    g._go_annotations(gene_file, code)
    print "processing data..........(takes time)"
    run = {'C':g._cellular_component, 'P':g._biological_process, 'F':g._molecular_function}
    ont = {'C':"Cellular Component", 'P':"Biological Process", 'F':"Molecular Function"}
    for i in ontology:
        i = i.split(":")
        print "working with %s ontology....."%ont[i[0]]
        objs[i[0]] = run[i[0]]()
        objs[i[0]]._species()
        objs[i[0]]._clustering(float(i[1]))
    return objs



def return_detail(result, geneA, geneB, detail):
    '''
    Formats the output for printing on screen or on file.
    '''
    domain_def = {'C':'Cellular Component', 'P':'Biological Process', 'F':'Molecular Function'}
    #r = "\nSemantic similarity between " + geneA + " and " + geneB + " is:\n\n"
    r = ''
    gene_pair = geneA + ',' + geneB
    for domain in result:
        #r += " " + domain_def[domain] + ": " + str(result[domain][0]) + "\n"
        r += gene_pair + "," + domain_def[domain] + ": " + str(result[domain][0]) + "\n"
        if detail:
            for data in result[domain][1]:
                r += "  GO id assigned to " + geneA + " is: " + data[0] + \
                     "\n  GO id assigned to " + geneB + " is: " + data[1] + \
                     "\n  LCA of assigned GO ids is: " + "|".join(result[domain][1][data]['lca']) + "\n\n"                  
    return r 
        
        

def calculate_semantic_similarity(objs, geneA, geneB, detail):
    '''
    Calls the function for calculating semantic similarity between
    genesA and genesB.
    '''
    result = {}
    for domain in objs:
        result[domain] = objs[domain]._semantic_similarity(geneA, geneB)
    return return_detail(result, geneA, geneB, detail)
        
    #cc._semantic_similarity('S000004065', 'S000001451')
    
