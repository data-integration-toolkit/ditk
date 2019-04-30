<h1> Bi-LSTM network from drug-drug interaction extraction</h1>
  This repo is based on the following paper-</br>
 •	 Drug-Drug Interaction extraction from biomedical text Using Long short term memory network</br>
 •	 Sunil Kumar Sahu, Ashish Anand. Drug-drug Interaction Extraction from BioMedical texts using Long short-term memory network.Journal of Biomedical Informatics(Volume 86, October 2018, pages 15-24),2018.
 
<h1>Original Code</h1>
https://github.com/sunilitggu/DDI-extraction-through-LSTM
<h1>Description</h1>
  This implementation handles the task of Relation extraction between drugs in a sentence.</br>
   •	Input - .txt file in the following format (Each column separated by \t)</br>
&nbsp;&nbsp;&nbsp;&nbsp;SENTENCE&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1_START_INDEX&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1_END_INDEX&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1TYPE&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_2&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_2_START_INDEX&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1_END_INDEX&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_2_TYPE&nbsp;&nbsp;&nbsp;&nbsp;RELATION</br>
   •	Output - .txt file with following contents</br>
&nbsp;&nbsp;&nbsp;&nbsp;SENTENCE&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_1&nbsp;&nbsp;&nbsp;&nbsp;ENTITY_2&nbsp;&nbsp;&nbsp;&nbsp;RELATION</br>
  It requires following files-</br>
   • train/dev/test files for training and prediction</br>
   • PubMed-w2v.bin from http://evexdb.org/pmresources/vec-space-models/</br>
 
<h1>Benchmark Dataset</h1></br>
 •	DDI 2013 (SemEval 2013 Task 9)</br>
 
 <h1>Evaluation metrics and Results</h1>
  •	metrics- precision, recall and f1</br>
 •	DDI 2013 - precision 0.5489</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;recall  -0.5255</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f1 - 0.5369 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f1 reported in the paper - 0.6939</br>

 <h1> Demo</h1>
  • <a href=https://github.com/samhithr/ditk/blob/develop/extraction/relation/RelationExtractionImpl/notebook/RelationExtraction_impl.ipynb>Jupyter Notebook</a></br>
  • <a href=https://youtu.be/eNgn_f3XNdg>Video</a></br>
