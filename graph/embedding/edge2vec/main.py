from transition import main as t_main
from edge2vec import main as e_main
from transition import parse_args as t_args
from edge2vec import parse_args as e_args
from eval import main as eval_main

from graph_embedding import graph_embedding

class edge2vec(graph_embedding):

    def read_dataset(self, file, options = None):
        return file

    def learn_embeddings(self,args, options = None):
        pass
            #Evaluate function is not a part of the code github given by the paper. Trying to implement for the metric calculation.

    def evaluate(self, options = {}):  #<--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()0!
        eval_main()
    """
            Evaluates the embedding by applying cosine similarity of 2 nodes which have an edge.
    
        """

    def save_model(self, args):
        t_main(args)
        """
        creates a transition model and saves it in a file
        """

    def load_model(self, args):
        e_main(args)
        """
        loads the model from the file to generate the embeddings and saves them in an output file
            """

    def main(self):
        file = "input.txt"
        data = self.read_dataset(file)
        self.save_model(t_args())
        self.load_model(e_args())
        self.evaluate()


if __name__=="__main__":
    edge = edge2vec()
    # edge.main()
