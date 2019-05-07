
from text.similarity.Siamese_LSTM.src.Siamese_LSTM_Similarity import Siamese_LSTM_Similarity
from decimal import Decimal

def main(input_file_path):
    instance = Siamese_LSTM_Similarity()
    instance.read_dataset(input_file_path)
    instance.load_model('../model/bestsem.p')
    sim_scores = []
    print("embedding and predicting")
    for i in range(len(instance.sentences_1)):
        score = instance.predict(instance.sentences_1[i],instance.sentences_2[i])
        sim_scores.append(score)
    file = open("../output/output_sim.txt", 'w')
    for score in sim_scores:
        score = str(Decimal(score).quantize(Decimal('0.00')))
        file.write(score)
        file.write('\n')

if __name__ == "__main__":
    main('../data/input.txt')




