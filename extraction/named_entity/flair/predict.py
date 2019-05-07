from flair.data import Sentence
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load("ner")

sentence1: Sentence = Sentence("George Washington went to Washington .")
tagger.predict(sentence1)

sentence2: Sentence = Sentence("Barack Obama was a very good president of USA .")
tagger.predict(sentence2)

# print("Analysing %s" % sentence1)
print("\nThe following NER tags are found: \n")
print(sentence1.to_tagged_string())
print(sentence2.to_tagged_string())

