train_transformer:
	bash train.sh transformer 1 5000000 0.001
test_transformer:
	bash eval.sh transformer 1 model_best.pt	
train_and_test_transformer: train_transformer test_transformer
	
train_rnn:
	bash train.sh rnn 0.5 1000000 0.003
test_rnn:
	bash eval.sh rnn 0.5 model_best.pt
train_and_test_rnn: train_rnn test_rnn

train_hybrid:
	bash train.sh hybrid 0.5 1000000 0.001
test_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt
train_and_test_hybrid: train_hybrid test_hybrid

transformer_all:
	bash train.sh transformer 0.5 1000000 0.001
	bash train.sh transformer 1 1000000 0.001
	bash train.sh transformer 2 1000000 0.001
	bash eval.sh transformer 0.5 model_best.pt
	bash eval.sh transformer 1 model_best.pt
	bash eval.sh transformer 2 model_best.pt
love: train_rnn train_hybrid
gen: 
	bash gen.sh