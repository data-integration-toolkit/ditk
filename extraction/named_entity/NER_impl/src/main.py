from named_entity.NER_impl.src.NER_impl import BiLSTMCRFNerImpl

filedict={"train":"..\\tests\\ner_test_input.txt","dev":"..\\tests\\ner_test_input.txt","test":"..\\tests\\ner_test_input.txt"}
obj=BiLSTMCRFNerImpl()
data=obj.read_dataset(filedict)
obj.train(data)
gt=obj.convert_ground_truth(data["test"])
pred=obj.predict(data)
metrics=obj.evaluate(pred,gt)
