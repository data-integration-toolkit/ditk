from mcdsat import mcdsat

def main():
	query_file = "test/sample_query.txt"
	views_file = "test/sample_views.txt"
	c2d_path = "c2d/c2d_linux"
	models = "dnnf-models/models"

	rewriting = mcdsat()
	rewriting.read_input(viewsFile=views_file, queryFile=query_file, c2d_path=c2d_path, models=models)
	mcds = rewriting.generate_MCDs()

	out = open("test/sample_output.txt", "w")
	out.write(str(mcds))
	out.close()

if __name__ == '__main__':
	main()