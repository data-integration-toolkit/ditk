from structured_gradient_tree_boosting import StructuredGradientTreeBoosting


def main(file_name):
    sgtb = StructuredGradientTreeBoosting()

    # read data
    ratio = (0.7, 0.2, 0.1)
    options = {}
    train_set, test_set, dev_set = sgtb.read_dataset(file_name, ratio, options)


    # train
    model = sgtb.train([train_set]+[dev_set])


    # save
    fileName = "./model/finalized_model.sav"
    sgtb.save_model(model, fileName)

    # load
    fileName = "./model/finalized_model.sav"
    model = sgtb.load_model(fileName)


    # predict
    result = sgtb.predict(model, test_set)

    # evaluation
    result = sgtb.evaluate(model, test_set)


if __name__ == "__main__":
    file_name = "./data/AIDA-PPR-processed.json"
    main(file_name)