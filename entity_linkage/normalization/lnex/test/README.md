Since the method uses twitter data the other common benhmarks for this method are not applicable, primarily becuase the modules has lot of tweet preprocessing functions


The input format is:
Text

The output format is list of output indicating entity boundary, entity extracted, normalized entity and metadata

The evaluation metric is also precision,recall and F1 score -> same as all members of group

The test cases are:

1. The predict method is giving correct results by checking non empty fields
2. The evaluate method is giving correct results by checking for precision, recall and F1 score

