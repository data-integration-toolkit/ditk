
import abc
import TextSemanticSimilarity



class OpenAI_Semantic_Similiarity(TextSemanticSimilarity):

    def analysis(self):
        import analysis.py

    def datasets(self):
        import datasets.py

    def text_utils(self):
        import text_utils.py

    def train(self):
        import train.py

    def utils(self):
        import utils.py

    def opt(self):
        import opt
