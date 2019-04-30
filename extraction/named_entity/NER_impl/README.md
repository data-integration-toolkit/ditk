<h1> Bi Directional LSTM CRF model for Named Entity Recognition</h1>
  This repo is based on the following paper-</br>
 •	 Neural Architectures for Named Entity Recognition</br>
 •	 Guillame Lample, Miguel Ballestros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. Neural Architectures for Named Entity Recognition. Proceedings of the 2016 conference of North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego, CA, USA,2016.
<h1>Original Code</h1>
https://github.com/guillaumegenthial/tf_ner/tree/master/models/chars_lstm_lstm_crf
<h1>Description</h1>
  This implementation handles the task of Named Entity Recognition from non tokenized sentences.</br>
   •	Input - .txt file in the following format</br>
&nbsp;&nbsp;&nbsp;&nbsp;WORD NER_tag</br>
   •	Output - .txt file with following contents</br>
&nbsp;&nbsp;&nbsp;&nbsp;WORD groundtruth_label predicted_label</br>
  It requires following files-</br>
   • train/dev/test files for training and prediction</br>
   • glove.840B.300d.txt from https://nlp.stanford.edu/projects/glove/</br>
 <h4>Architecture</h4>

1. GloVe 840B vectors
2. Chars embeddings
3. Chars bi-LSTM
4. Bi-LSTM
5. CRF

<h1>Benchmark Dataset</h1></br>
 •	CoNLL 2003</br>
 •	Ontonotes 5.0
 <h1>Evaluation metrics and Results</h1>
  •	metrics- precision, recall and f1</br>
 •	CoNLL 2003 - precision 0.9041</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;recall  -0.9056</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f1 - 0.9049 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f1 reported in the paper - 0.9094</br>
 •	Ontonotes - precision -0.7642</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;recall- 0.7389</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f1 - 0.7513 </br>
 <h1> Demo</h1>
  • <a href=https://github.com/samhithr/ditk/blob/develop/extraction/named_entity/NER_impl/notebook/NER_impl.ipynb>Jupyter Notebook</a></br>
  • <a href=https://youtu.be/mGRW-NmnUAE>Video</a></br>
  
