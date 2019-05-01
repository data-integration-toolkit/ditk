
import sensim
from sensim import sensim
def main():
    filename = "data/training_data.csv"
    obj=sensim()
    sentences1,sentences2,all_sentences,similarity_score=obj.read_dataset(filename)
    obj.train(sentences1, sentences2, all_sentences, similarity_score)
    obj.evaluate("training_features.csv")
main()
