from cdma_model import CDMAModel
import matplotlib.pyplot as plt

def main(input_file_path):

    model = CDMAModel()
    model.load_model("target_model/model_weights", "target_model/") #pretrained model
    model.read_dataset(None, input_file_path)
    loss_hist, train_acc_hist = model.train(None)
    """
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    # loss_hist_ = loss_hist[1::100]  # sparse the curve a bit
    plt.plot(loss_hist, '-o')
    plt.xlabel('Iteration')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_hist, '-o', label='Training')
    plt.xlabel('Epoch')
    plt.axis([0, 5, 0, 100])
    plt.legend(loc='lower right')
    #plt.gcf().set_size_inches(15, 12)
   
    plt.savefig('ontonotes-nw.png')
    #plt.show()
    """
    p, f1, r = model.evaluate(None, None)
    #output_path = model.predict(input_file_path+"/test")
    output_path = model.predict("ner_test_input.txt") # G6 NER test input_sample
    return output_path


if __name__ == "__main__":
    path = "datasets/ontonotes-nw" # input training path
    out_path = main(path)
    print("Predict_output_path: ", out_path)
