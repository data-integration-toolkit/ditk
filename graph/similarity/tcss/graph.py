'''
Created on 2010-07-19

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

class Graph(Exception):
    '''
    Graph class consists of general graph manipulation functions. 
    '''


    def __init__(self):
        '''
        Initialises graph dictionary, node_cont, edge_count and 
        flip dictionary variables. 
        
        graph variable stores graph information in parent->child 
        format with nodes as keys of dictionary and set of child 
        nodes as values. 
        
        flip variable stores graph information in child->parent
        format with nodes as keys of dictionary and set of parent 
        nodes as values.
        '''
        super(Graph, self).__init__()
        self.graph = {}
        self.node_count = 0
        self.edge_count = 0
        self.flip = {}

    
        
    def _add_node(self, node):
        '''
        Function takes a node (string/int) as an input and add it 
        to the graph/flip if it is not present in the graph. 
        '''
        if node not in self.graph:
            self.graph[node] = set()
            self.flip[node] = set()
            self.node_count += 1
        
            
    def _add_edge(self, parent, child): 
        '''
        Function takes a parent and child node (string/int) as an 
        input and add the edge to the graph/flip if it is not present
        in the graph. 
        '''
        if parent not in self.graph: raise Exception('node not found')
        if child not in self.graph: 
            self._add_node(child)
        self.graph[parent].add(child)
        self.flip[child].add(parent)
        self.edge_count += 1
    
            
    def _remove_node(self, node):
        '''
        Function takes a node (string/int) as an input and removes 
        it from the graph/flip if it is present in the graph. 
        '''   
        self.edge_count -= len(self.graph[node])
        self.graph.pop(node)
        self.flip.pop(node)
        self.node_count -=  1
    
    
    def _remove_edge(self, parent, child):
        '''
        Function takes an edge (parent->child) (string/int) as an 
        input and removes it from the graph/flip if it is present in 
        the graph. 
        '''
        if parent in self.graph and child in self.graph[parent]: 
            self.graph[parent].remove(child)
            self.flip[child].remove(parent)
            self.edge_count -= 1
        
        
    def _update_graph(self, graph):
        '''
        Function takes a graph (dictionary) as an input and updates
        the current graph.
        '''
        for node in graph:
            self._add_node(node)
            for child in graph[node]:
                self._add_node(child)
                self._add_edge(node, child)
            
    
    def _depth_first_search(self, node = None, target = None, visited = None, rev = False):
        '''
        Function takes a node, target (optional), visited (dict, optional), 
        rev (bool, optional) and performs a depth first search. Depending 
        upon the input it could return a graph (dictionary) starting from 
        node or find a target in current graph. If rev == True then the
        search will be performed on flip graph. 
        '''
        if rev: graph = self.flip
        else: graph = self.graph
        if node == None: raise Exception("incomplete arguments: provide graph/node")
        if node not in graph: raise Exception("invalid node: node not present in graph")
        if visited == None: visited = {}
        if target and node == target: return True
        visited[node] = set(list(graph[node])[:])
        for child in graph[node]:
            if child not in visited:
                visit = self._depth_first_search(child, target, visited, rev)
                if target and visit == True: return visit
        if target: return False 
        else: return visited
        
        
    '''def _flip_graph_edges(self):
        obsolete function: not used any more
        new_graph = Graph()
        for node in self.graph:
            new_graph._add_node(node)
            for child_node in self.graph[node]:
                new_graph._add_node(child_node)
                new_graph._add_edge(child_node, node)
        return new_graph'''
    
    
    def _remove_unwanted_edges(self):
        '''
        Removes edges of the type: if edges a->b->c and a->c exist then
        a->c is removed.
        '''
        unwanted = {}
        for node in self.graph:
            children = list(self.graph[node])[:]
            for child_i in children:
                for child_j in children:
                    if child_i != child_j:
                        if (child_i, child_j) in unwanted:
                            self._remove_edge(node, child_j)
                        else:
                            if self._depth_first_search(child_i, child_j):
                                self._remove_edge(node, child_j)
                                unwanted[(child_i, child_j)] = True
                                

                            
                            
                        
                
                
        
        
        