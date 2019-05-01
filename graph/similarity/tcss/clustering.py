'''
Created on 2010-07-22

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

import math
import copy

class Clustering():
    '''
    Clustering class implement functions for performing topological
    clustering and calculating entropy based on clustering. 
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(Clustering, self).__init__()
    
        
        
    def _entropy(self, a, b):
        '''
        Given a and b (float/int) it returns the information content or
        Shanon's entropy.
        '''
        return -math.log10(a / float(b)) 
    
     
        
    def _meta_graph_nodes(self, cutoff):
        '''
        Given a topology cutoff the function find the terms for higher
        level graph. If the entropy values of parent - child terms is
        in close proximity of each other then the child term is removed.
        '''
        meta_terms = {}
        for node in self.graph:
            children = set(self._depth_first_search(node).keys())
            ICT = self._entropy(len(children), self.node_count)
            if ICT <= cutoff:
                meta_terms[node] = {'ICT':ICT, 'children':children}
        terms = copy.deepcopy(meta_terms)
        for term1 in terms:
            for term2 in self.flip[term1].intersection(terms):
                if terms[term2]['ICT'] != 0 and \
                   terms[term1]['ICT'] / terms[term2]['ICT'] < 1.2:
                    meta_terms.pop(term1)
                    break
        return meta_terms
        
        
        
    def _sub_graph_nodes(self, meta_graph, meta_terms):
        '''
        Given higher level graph and term set, the function find the
        terms for sub-graphs. 
        '''
        sub_terms = {}
        for term in meta_graph.graph:
            sub_terms[term] = meta_terms[term]['children']
            for node in meta_graph.graph[term]:
                sub_terms[term] = sub_terms[term].difference(meta_terms[node]['children'])
        return sub_terms
    
            
            
    def _entropy_graph(self, graph, clusid):
        '''
        Given a graph the function calculates the normalised entropy of
        each node present in that graph.
        '''
        maxm = 0
        for node in graph.graph:
            self.go_annotations[node]['cluster'][clusid] = {}
            self.go_annotations[node]['cluster'][clusid]['entropy'] = self._entropy(len(graph.go_annotations[node]['gene']),\
                                          len(graph.go_annotations[self.goid]['gene']))
            if self.go_annotations[node]['cluster'][clusid]['entropy'] > maxm:
                maxm = self.go_annotations[node]['cluster'][clusid]['entropy']
        
        if maxm == 0:
            return
        
        for node in graph.graph:
            self.go_annotations[node]['cluster'][clusid]['entropy'] = self.go_annotations[node]['cluster'][clusid]\
                                                                                         ['entropy'] / maxm  
                                                                                         
    
    
    def _ancestor_terms(self, term_set, term):
        '''
        Given a set of terms and term, the function finds the ancestors
        of the given term in the term set.
        '''
        for node in term_set:
            #self.go_annotations[node]['cluster'][term]['ancestors'] = \
                                            #set(sub_graph[term]._depth_first_search(node = node, rev = True).keys())
            self.go_annotations[node]['cluster'][term]['ancestors'] = self.go_annotations[node]['ancestors']. \
                                                                            intersection(term_set)
                    
            
            
               
            
        
            
                 
            
        