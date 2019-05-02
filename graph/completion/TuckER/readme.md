TuckER: Tensor Factorization for Knowledge Graph Completion 
Ivana Balazevi, Carl Allen, Timothy Hospedales

Input: 
E1 Relation E2
Where E1 is Subject Entity or head
E2 is Object Entity or tail

Output:
Assigns a probability to each triple in range [0,1]. If probability is less than 0.5, then E1 and E2 are not related to each other by relation R, if probability is greater than 0.5, then E1 and E2 are related to each other by relation R.

Approch:
The aim is to learn a scoring function φ that
assigns a score s = φ(es, r, eo) ∈ R to each triple, 
indicating whether that triple is true or false.
We use sigmoid function to calculate the
Probablity that the given triple true or false 
So it can work for non-linear functions as well.
We make closed-world assumption. It means that if any triple is incomplete or it has some missing values, we consider it as negative example. 

 


where es, eo ∈ R de are the rows of E representing the subject and object entity embedding vectors, wr ∈ R dr the rows of R representing the relation embedding vector, W ∈ R de×dr×de is the core tensor of Tucker decomposition and ×n is the tensor product along the n-th mode. We apply logistic sigmoid to each score φ(es, r, eo) to obtain the predicted probability p = σ(φ(es, r, eo)) of a triple being true. Visualization of the TuckER model architecture can be seen in Figure 1.

Environment:
Python 3.6.6
Numpy 1.14.5
CUDA (>=8.0)
Nvidia Driver and Topend Nvidia GPU (required)

Execute the code: 
/*for WN18 Dataset. Change the value in dataset to execute code for another dataset */
CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18 --num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1

Benchmark Datasets:
FB15k
WN18
FB15k-237
WN18RR

Evaluation Metrics:
MRR
hits@10

Results:
Datasets	MRR	hits@10
FB15k	0.734	0.912
WN18	0.897	0.886
FB15k-237	0.387	0.524
WN18RR	0.531	0.523



Link to Youtube video: https://youtu.be/Khh98gUfp4Q


