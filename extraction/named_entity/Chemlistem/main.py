from DITKChemlistem import DITKChemlistem

def main(file):
    dataset = {"train": file}
    model = DITKChemlistem()
    dataset = model.read_dataset(dataset, "ditk")

    model.train(dataset["train"])
    predictions = model.predict(dataset["test"])

    ground_truth = model.convert_ground_truth(dataset["test"])
    assert(len(ground_truth) == len(predictions))
    print(model.evaluate(predictions,ground_truth))
    return model.write_ditk_output(predictions, dataset["train"])
