train_transformer:
	bash train.sh transformer 1 5000000 0.001
test_transformer:
	bash eval.sh transformer 1 model_best.pt	
train_and_test_transformer: train_transformer test_transformer
	
train_rnn:
	bash train_rnn.sh 0 0
test_rnn:
	bash eval.sh rnn model_1.pt 0.5m_rnn.json
train_and_test_rnn: train_rnn test_rnn

train_hybrid:
	bash train_hybrid.sh 0 0
test_hybrid:
	bash eval.sh hybrid model_1.pt 0.5m_hybrid.json
train_and_test_hybrid: train_hybrid test_hybrid
