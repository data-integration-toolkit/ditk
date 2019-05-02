import sys
import os

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from entity_linkage.typing.oneShotRelationalLearning.one_shot_relational_learning import OneShotRelationalLearning

def main(input_file_path) :
	one_shot = OneShotRelationalLearning()
	train_dir = one_shot.read_dataset([input_file_path])
	#one_shot.train(train_dir)
	#return one_shot.predict(train_dir)

if __name__ == '__main__':
	main("NELL_min.kb")
