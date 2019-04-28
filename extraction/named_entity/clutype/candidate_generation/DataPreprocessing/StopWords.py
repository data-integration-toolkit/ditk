import inspect, os


class StopWords:
    def __init__(self):
        candidate_generation_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
        path = os.path.join(candidate_generation_dir, 'stopwords/en.txt')
        f = open(path, 'r')
        self.stop_words = set([line.strip() for line in f])

    def isStopWord(self, word):
        if word in self.stop_words:
            return True
        return False
