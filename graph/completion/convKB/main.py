from conv_kb import ConvKB

def main(file_name):
    convkb = ConvKB()

    options = {}
    options["split_ratio"] = (0.7, 0.25, 0.05)
    options["embedding_dim"] = 50
    options["batch_size"] = 128
    train, dev, test = convkb.read_dataset(file_name, options=options)


    # train
    options = {}
    options["num_epochs"] = 6
    options["save_step"] = 5
    model = convkb.train(train, options=options)

    # predict
    options = {}
    options["model_index"] = 5
    result = convkb.predict(test, options=options)

    # evaluation
    options = {}
    options["model_index"] = 5
    dic = convkb.evaluate(test, options=options)

    return "./data/"


if __name__ == "__main__":
    file_name = "./data/sample_input.txt"
    main(file_name)