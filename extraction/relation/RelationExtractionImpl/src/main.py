from relation.RelationExtractionImpl.src.RelationExtraction_impl import DDIExtractionImpl

obj=DDIExtractionImpl()
input = {"train": "..\\tests\\relation_extraction_test_input.txt",
                 "dev": "..\\tests\\relation_extraction_test_input.txt",
                 "test": "..\\tests\\relation_extraction_test_input.txt"}
obj.train(input)
sent_pred = obj.predict()
scores = obj.evaluate(sent_pred[1], sent_pred[2])