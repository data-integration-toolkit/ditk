from DITKD3NER import DITKD3NER

def main(file):
    model = DITKD3NER()
    dataset = {"train": file}
    dataset_converted, dataset_regular = model.read_dataset(dataset, 'ditk')
    model.train(dataset_converted)
    predictions = model.predict(dataset_converted["test"])
    ground_truth = model.convert_ground_truth(dataset_regular["test"])
    print(model.evaluate(predictions,ground_truth))
    return model.write_ditk_output(predictions,dataset_regular["train"])


