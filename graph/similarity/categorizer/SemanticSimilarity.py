# -*- coding: ms949 -*-

import math


CHILD = 'C'
PARENT = 'P'

class SEMANTIC_SIMILARITY:
    
    index = None
    prob = None
    
    def __init__(self, index_file = 'go_index.txt', prob_file = 'go_prob.txt'):
        
        # index[goid] = { PARENT: [1,2,3..], CHILD: [1,2,3,...] }
        self.index = self.__loadIndex(index_file)
        
        # prob[goid] = float
        self.prob = self.__loadProbability(prob_file)
        
    def __loadProbability(self, fname):
        prob = {}
        f=open(fname,'r')
        
        for s in f.readlines():
            s = s.replace('\n','')
            if len(s) == 0: continue
            
            x = s.split('\t')
            goid = x[0]
            pr = float( x[1] )
            prob[goid]=pr
            
            
        f.close()
        
        return prob
        
    
    def __removeBlankElement(self, your_list):
        r = []
        for y in your_list:
            y = y.strip()
            if len(y)>0:
                r.append(y)
                
        return r
    
    def __loadIndex(self, fname):
            

        
        index = {}
        
        f=open(fname,'r')
        
        for s in f.readlines():
            s = s.replace('\n','')
            if len(s) == 0: continue
            
            x = s.split('\t')
            goid = x[0] # go id
            pids = self.__removeBlankElement( x[1].split(',') ) # parents
            cids = self.__removeBlankElement( x[2].split(',') ) # children
            
            index[goid] = {
                PARENT: pids,
                CHILD: cids
                }
            

            
        f.close()
    
        return index
    
    def __dist(self, goid1, goid2):

        
        c1 = self.prob[goid1]
        c2 = self.prob[goid2]
        
        d = abs( - math.log(c1) + math.log(c2) )
        return d
        
    def __alpha(self, goid):
        return -math.log( self.prob[goid] )
    
    def __beta1(self, goid):

        
        found_leaf = None
        found_leaf_log_p = 0
        
        for g in self.index[goid][CHILD]:
            
            d = - math.log( self.prob[g] )
            if d > found_leaf_log_p:
                found_leaf = g
                found_leaf_log_p = d
        
        d = 0.0
        if found_leaf is not None:
            d = self.__dist(goid, found_leaf)

        return d
        
    def __beta(self, goid1, goid2):
        beta = ( self.__beta1(goid1) + self.__beta1(goid2) ) / 2.0
        return beta
    
    def __gamma(self, goid1, goid2, common_parent):
        gamma = self.__dist(goid1, common_parent) + self.__dist(goid2, common_parent)
        return gamma
        
    
    def __getCommonParents(self, goid1, goid2):

        
        r = []
        g1 = self.index[goid1][PARENT] + [ goid1 ]
        
        g2 = [ goid2 ] #--> strict relation
        #g2 = self.index[goid2][PARENT] + [ goid2 ]
        
        for g in g1:
            if g in g2:
                r.append(g)
                
        return r
    
    def __hrss(self, goid1, goid2, common_parent):
        
        a = self.__alpha(common_parent)
        b = self.__beta(goid1, goid2)
        g = self.__gamma(goid1, goid2, common_parent)
        
        h = 0.0
        if a != 0.0:
            h = 1.0/(1.0+g) * a/(a+b)

        return h


    def __getMostInformativeParent(self, common_parents):

        #return common_parents
    
        # return only one common parent with the lowest prob.
        r = []

        prob = 1.0
        go = None

        for c in common_parents:
            if go is None:
                go = c
                prob = self.prob[go]
                r = [go]
            else:
                if self.prob[go] < prob:
                    go = c
                    prob = self.prob[go]
                    r = [go]

        return r







    def getSimilarity(self, goid1, goid2):
        '''
        goid1 should be a child term of goid2
        '''

        common_parents = self.__getCommonParents(goid1, goid2)
        informative_parent = self.__getMostInformativeParent(common_parents)


        dist = []
        
        for c in informative_parent:
            d = self.__hrss(goid1, goid2, c)
            dist.append(d)
            

        if len(dist) == 0:
            max_dist = -1.0 # if there is no link to GO terms in the definition file
        else:
            max_dist = max(dist)
        
        return max_dist
    
    
if __name__ == '__main__':
    # test
    
    
    g2 = 'GO:0090514' # child, G10
    g1 = 'GO:0015807' # parent, G2
    
    s = SEMANTIC_SIMILARITY()
    print s.getSimilarity(g2, g1)
    
    g2 = 'GO:0090514' # child, G10
    g1 = 'GO:0015801' # parent, G3
    
    s = SEMANTIC_SIMILARITY()
    print s.getSimilarity(g2, g1)
  
    g2 = 'GO:0051939' # child, G8
    g1 = 'GO:0043090' # parent, G4
    
    s = SEMANTIC_SIMILARITY()
    print s.getSimilarity(g2, g1)
    
    
    g2 = 'GO:0051939' # child, G8
    g1 = 'GO:0015812' # parent, G4
    
    s = SEMANTIC_SIMILARITY()
    print s.getSimilarity(g2, g1)        