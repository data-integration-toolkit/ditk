It can use all the functions in the graph_completion template except test_output_facts() function. 

The test_output_facts() in the template compares input and output triples and expects to have same number of unique entities and relations.

However, my read_dataset() split input file to three portions and convert the input format to dictionary, so that it is hard to have same number of items for input and output. 

Also since completing triples from either head or tail results in different number of unique entities and relations in input and output, I was not able to test on test_output_facts().