'''
Created on 2010-07-26

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

class SemanticSimilarity(object):
    '''
    SemanticSimilarity class implement functions for calculating 
    topology based semantic similarity between genes.  
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(SemanticSimilarity, self).__init__()
        
        
    def _lowest_common_ancestor(self, termA, termB, cluster):
        '''
        Given GO terms A and B and the graph to which they belong
        the function finds the lowest common ancestor i.e. the 
        ancestor of terms A and B with highest entropy. 
        '''
        lca = {}
        for term in self.go_annotations[termA]['cluster'][cluster]['ancestors'] \
                            .intersection(self.go_annotations[termB]['cluster'][cluster]['ancestors']):
            entropy = self.go_annotations[term]['cluster'][cluster]['entropy']
            if entropy not in lca:
                lca[entropy] = set()
            lca[entropy].add(term) 
        return max(lca), lca[max(lca)]
        
        
        
    def _semantic_similarity(self, geneA, geneB):
        '''
        Given genes A and B the function calculates the semantic similarity
        between them. It uses gene_annotations and go_annotaions variables.
        '''
        sem_sim = {}
        if geneA not in self.gene_annotations or geneB not in self.gene_annotations:
            if geneA not in self.gene_annotations:
                print("%s not found in annotations"%geneA)
            if geneB not in self.gene_annotations:
                print("%s not found in annotations"%geneB)
            return None, None
            
        for termA in self.gene_annotations[geneA]:
            for termB in self.gene_annotations[geneB]:
                for clusA in self.go_annotations[termA]['cluster']:
                    for clusB in self.go_annotations[termB]['cluster']:
                        if clusA != 'meta' and clusB != 'meta' and clusA == clusB:
                            value, lca = self._lowest_common_ancestor(termA, termB, clusA)
                            if value not in sem_sim:
                                sem_sim[value] = {}
                            sem_sim[value][(termA, termB)] = {'lca':lca, 'clusA':clusA, 'clusB':clusB}
                        elif clusA != 'meta' and clusB != 'meta' and clusA != clusB:
                            value, lca = self._lowest_common_ancestor(clusA, clusB, 'meta')
                            if value not in sem_sim:
                                sem_sim[value] = {}
                            sem_sim[value][(termA, termB)] = {'lca':lca, 'clusA':clusA, 'clusB':clusB}
        return max(sem_sim), sem_sim[max(sem_sim)]
                            
                            
                
            
