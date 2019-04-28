import inspect, os


class PPV:
    def __init__(self):
        candidate_generation_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
        path = os.path.join(candidate_generation_dir, 'EntityExtraction/PPV/ppv.txt')
        self.words = set([word.strip() for word in open(path, 'r')])
        self.replacement = "ppv"

    def collapse(self, sentence):
        new_sentence = []
        for phrase in sentence:
            new_phrase = []
            for word in phrase:
                if word in self.words:
                    new_phrase.append(self.replacement)
                else:
                    new_phrase.append(word)
            new_sentence.append(new_phrase)
        return new_sentence
