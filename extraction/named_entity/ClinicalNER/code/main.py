from ClinicalNER import clinicalNER
NER = clinicalNER()

train_txt_path = "data/converted/CoNLL_train.txt"
train_con_path = "data/converted/CoNLL_train.con"
NER.read_dataset("data/origin/CoNLL.train.txt", train_txt_path, train_con_path)

test_txt_path = "data/converted/CoNLL_test.txt"
test_con_path = "data/converted/CoNLL_test.con"
NER.read_dataset("data/origin/CoNLL.test", test_txt_path, test_con_path)

model_path = "models/CoNLL.model"
NER.train(train_txt_path, train_con_path, model_path)

prediction_dir = "data/test_predictions/"
prediction_path = NER.predict(model_path, test_txt_path, prediction_dir)

NER.evaluate(prediction_path, test_con_path)

output_path = "data/output/CoNLL_test.output"
NER.output(test_txt_path, test_con_path, prediction_path, output_path)