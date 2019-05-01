'''
Created on 2010-07-20

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

from graph import Graph
from clustering import Clustering
from parser import Parser
from semantic import SemanticSimilarity

class GOGraph(Graph, Clustering, Parser, SemanticSimilarity):
    '''
    Class GOGraph implement graph functions specific for
    gene ontology and inherits Graph, Clustering, Parser,
    SemanticSimilarity classes.
    '''


    def __init__(self):
        '''
        Initialises go_annotations, goid, and gene_annotations 
        variables.
        
        go_annotations variable contains detailed information of
        GO terms and also stores the results of data processing to
        be used for calculating semantic similarity.
        {GO_term:{'name':"",'domain':"", 'gene':set(), 'ancestors':set(),
         'clusters':{'cluster_id':"", 'entropy':float(), 'ancestors':set()
         }}}
        
        goid takes one of the following values: 
        GO:0005575 (cellular_component)
        GO:0008150 (biological_process)
        GO:0003674 (molrcular_function)
        
        gene_annotations variable contains GO terms assigned to genes. 
        {gene:set(GO_term)}
        '''
        super(GOGraph, self).__init__()
        self.go_annotations = {}
        self.goid = ''
        self.gene_annotations = {}
          
        
        
    def _create_graph_from_node_set(self, node_set):
        '''
        Given a set of nodes create a graph from it. GOGraph object is
        returned.
        '''
        new_graph = GOGraph()
        for node1 in node_set:
            new_graph._add_node(node1)
            for node2 in node_set:
                if node1 != node2 and self._depth_first_search(node = node1, target = node2):
                    new_graph._add_edge(node1, node2)
        new_graph._remove_unwanted_edges()
        new_graph.go_annotations = self.go_annotations
        new_graph.goid = self.goid
        return new_graph
                                
                    
                    
    def _create_graph_from_node(self, node):
        '''
        Given a node create use depth first search to create a new graph.
        GOGraph object is returned.
        '''
        new_graph = GOGraph()
        #new_graph.go_annotations = self.go_annotations
        new_graph.goid = node
        new_terms =  self._depth_first_search(node)
        new_graph._update_graph(new_terms)
        for term in new_terms:
            new_graph.go_annotations[term] = self.go_annotations[term]
        return new_graph
                                
               
                                
    def _species(self):
        '''
        Prune the current graph to have nodes with genes annotated to them for
        a particular species. For this genes are propagated first to parent
        terms.
        '''
        for node in self.flip:
            parent_terms = self._depth_first_search(node = node, rev = True).keys()
            self.go_annotations[node]['ancestors'] = set(parent_terms)
            for parent in parent_terms:
                self.go_annotations[parent]['gene'] = self.go_annotations[parent]['gene'] \
                                                    .union(self.go_annotations[node]['gene'])
        for term in self.graph.keys():
            if len(self.go_annotations[term]['gene']) == 0:
                self.go_annotations.pop(term)
                for parent in list(self.flip[term])[:]:
                    self._remove_edge(parent, term)
                self._remove_node(term)    
                
                
                
    def _gene_annotations(self):
        '''
        Assign genes to the most specific node among all the nodes it has been 
        originally annotated to in a graph. 
        '''
        for term in self.go_annotations:
            for gene in self.go_annotations[term]['gene']:
                if gene not in self.gene_annotations:
                    self.gene_annotations[gene] = set()
                self.gene_annotations[gene].add(term) 
            
        for gene in self.gene_annotations:
            go_list = list(self.gene_annotations[gene])[:]
            for term1 in go_list:
                for term2 in go_list:
                    if term1 != term2 and \
                        term1 in self.go_annotations[term2]['ancestors'] \
                        and term1 in self.gene_annotations[gene]:
                        self.gene_annotations[gene].remove(term1)
                        
            
                
                    
    def _clustering(self, cutoff):
        '''
        Function takes the topology cutoff as the input and call other functions
        to create higher/lower level graphs and entropy calculation. go_annotation
        and gene_annotation variables are updated will all the information.
        '''
        meta_terms = self._meta_graph_nodes(cutoff)
        meta_graph = self._create_graph_from_node_set(meta_terms)
        self._entropy_graph(meta_graph, 'meta')
        self._ancestor_terms(meta_terms, 'meta')
        sub_terms = self._sub_graph_nodes(meta_graph, meta_terms)
        sub_graph =  {}
        for term in sub_terms:
            sub_graph[term] = self._create_graph_from_node_set(sub_terms[term])
            self._entropy_graph(sub_graph[term], term)
            self._ancestor_terms(sub_terms[term], term)
        self._gene_annotations()    
        
           
                    
    def _cellular_component(self):
        '''
        Creates a cellular component gene ontology graph.
        '''
        return self._create_graph_from_node('GO:0005575')
    
    
    
    def _biological_process(self):
        '''
        Creates a biological process gene ontology graph.
        '''
        return self._create_graph_from_node('GO:0008150')
    
    
    
    def _molecular_function(self):
        '''
        Creates a molecular function gene ontology graph.
        '''
        return self._create_graph_from_node('GO:0003674')
    
    
    
                    
                    
                
            
                
                                
                