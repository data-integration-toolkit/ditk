from simple import SimplE

def main(input_file_path):
    '''Run SimplE
    
    Assumes datasets are at ./<input_file_path>/<dataset_name>
    and are already split into ./<input_file_path>/<dataset_name>/{train.txt, test.txt, valid.txt}
    ''' 
    s = SimplE(input_file_path)
    datasets = ["fb15k", "wn18"]
    training, test, valid = s.read_dataset(datasets[0])
    s.train(training)
    new_triples = s.predict(test)
    print(s.evaluate(valid, ["mrr", "hits10"]))

    output_file_path = "new_triples.txt"

    with open(output_file_path, "w") as f:
        for triple in new_triples:
            f.write("{0} {1} {2}\n".format(*triple))

    return output_file_path


if __name__ == "__main__":
    main("../data")
