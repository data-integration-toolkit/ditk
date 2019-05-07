from ClinicalRE import clinicalRE

RE = clinicalRE()

coverted_txt_path = "data/converted/SemEval.txt"
RE.read_dataset("data/origin/SemEval.txt", coverted_txt_path)

#label_dict = {'other': 0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}
#label_dict = {'other': 0, 'int': 1, 'advise': 2, 'effect': 3, 'mechanism': 4}
label_dict = {'Other': 0, 'Message-Topic(e2,e1)': 1, 'Cause-Effect(e1,e2)': 2, 'Component-Whole(e2,e1)': 3, 'Entity-Origin(e2,e1)': 4, 'Member-Collection(e2,e1)': 5, 'Message-Topic(e1,e2)': 6, 'Instrument-Agency(e1,e2)': 7, 'Product-Producer(e1,e2)': 8, 'Instrument-Agency(e2,e1)': 9, 'Entity-Destination(e1,e2)': 10, 'Content-Container(e2,e1)': 11, 'Entity-Origin(e1,e2)': 12, 'Member-Collection(e1,e2)': 13, 'Entity-Destination(e2,e1)': 14, 'Component-Whole(e1,e2)': 15, 'Product-Producer(e2,e1)': 16, 'Cause-Effect(e2,e1)': 17, 'Content-Container(e1,e2)': 18}

output_file = "data/output/SemEval.output"
RE.train(coverted_txt_path, label_dict, output_file)
