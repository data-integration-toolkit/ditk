from ReBran import ReBran

def main(input_file_path):
    input = {'train': input_file_path,
            'test': input_file_path,
            'dev': input_file_path}

    my_model = ReBran()
    data = my_model.read_dataset(file_names=input)

    my_model.train(train_data=data)

    pred_vals = my_model.predict(test_data=data['test'])
    my_model.evaluate(prediction_data=pred_vals)
    output_file_path = my_model.data_dir+"/output.txt"

    return output_file_path
