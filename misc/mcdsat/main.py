from mcdsat import mcdsat

def main():
	query_file = "examples/query_0.txt"
	views_file = "examples/views_0.txt"
	c2d_path = "c2d/c2d_linux"
	models = "dnnf-models/models"

	rewriting = mcdsat()
	rewriting.read_input(viewsFile=views_file, queryFile=query_file, c2d_path=c2d_path, models=models)
	rewriting.generate_MCDs()

if __name__ == '__main__':
	main()