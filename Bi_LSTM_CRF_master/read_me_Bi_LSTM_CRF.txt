This Folder contains The KP dataset for Bi_LSTM_CRF model, code of Bi_LSTM_CRF, pretrained models and evaluating results .

###################Datasets###########################

Original KP data set:  
	kp20k: 527K training data set; 20k validation set; 20k test set.
	kp20k_small: small samples from kp20k, used for debugging.


KP_document:
	#### Intermidiate data after preprocessing for document-based model (generated by ./utils/preprocessing_doc.py). ###
	valid_data.pkl
	train_data.pkl
	test_data.pkl
	dictionary.pkl : saving word2id and id2word dictioanry.
	Embedding.pkl  : Pretrained Glove embedding matrix for words in our vocabulary. 



KP_sentence: 
	#### Intermidiate data after preprocessing for sentence-based model. (generated by ./utils/preprocessing_sent.py).###
	valid_data.pkl
	train_data.pkl
	test_data.pkl
	dictionary.pkl : saving word2id and id2word dictioanry.
	Embedding.pkl  : Pretrained Glove embedding matrix for words in our vocabulary. 


###################Code of Bi_LSTM_CRF#######################################################################################

(1) Document-based model:
	training: run main_doc.py  , generate pretrained model.

	testing:  run test_doc.py   , generate evaluation results.

	plotting: run plotting.py for the evaluation results


(2) Sentence-based model:
	training: run main_sent.py  , generate pretrained model.

	testing:  run test_sent.py   , generate evaluation results.

	plotting: run plotting.py for the evaluation results

##################### Pretrained model ####################################################################################
doc-based model is saved in ./checkpoint/doc

sent-based model is saved in ./checkpoint/sent


##################### Intermediate results after testing and for plotting #################################################

(1) Document-based model
	train_doc.csv : batch average precision, recall and f1-score on part of training set.
	valid_res.pkl : batch average precision, recall and f1-score on part of validation set.
	test_res.pkl  : batch average precision, recall and f1-score on part of test set.

(2)Sentence-based model
	train_sent.csv :     batch average precision, recall and f1-score on part of training set.
	valid_res_sent.pkl : batch average precision, recall and f1-score on part of validation set.
	test_res_sent.pkl  : batch average precision, recall and f1-score on part of test set.





