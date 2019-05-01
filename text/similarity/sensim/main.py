
import sensim
from sensim import sensim
def main():
    filename = "C:\\Users\\sanji\\Desktop\\sts-train.csv"
    obj=sensim()
    sentences1,sentences2,all_sentences,similarity_score=obj.read_dataset(filename)
    print(sentences1[0])
    obj.train(sentences1, sentences2, all_sentences, similarity_score)
    obj.evaluate("C:\\Users\\sanji\\Desktop\\training_features.csv")
main()