import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from graph.completion.convKB.conv_kb import ConvKB

def main(file_name):
    convkb = ConvKB()

    options = {}
    options["split_ratio"] = (0.7, 0.25, 0.05)
    options["embedding_dim"] = 100
    options["batch_size"] = 128
    train, dev, test = convkb.read_dataset(file_name, options=options)


    # train
    options = {}
    options["num_epochs"] = 21
    options["save_step"] = 20
    model = convkb.train(train, options=options)

    # predict
    options = {}
    options["model_index"] = 20
    result = convkb.predict(test, options=options)

    # evaluation
    options = {}
    options["model_index"] = 20
    dic = convkb.evaluate(test, options=options)

    ditk_path = ""
    for path in sys.path:
        if "ditk" in path and not "graph" in path:
            ditk_path = path
    output_path = ditk_path + '/graph/completion/convKB/result/output.txt'  
    return output_path


if __name__ == "__main__":
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path and not "graph" in path:
            ditk_path = path
    file_name = ditk_path + "/graph/completion/convKB/data/sample_input.txt"
    main(file_name)