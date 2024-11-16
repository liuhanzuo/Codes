train_transformer:
	bash train.sh transformer 1 5000000 0.0001 64
test_transformer:
	bash eval.sh transformer 0.5 model_best.pt 5000000
train_and_test_transformer: train_transformer test_transformer
	
train_rnn:
	bash train.sh rnn 0.5 5000000 0.0003 1024
test_rnn:
	bash eval.sh rnn 0.5 model_best.pt 5000000
train_and_test_rnn: train_rnn test_rnn

train_hybrid:
	bash train.sh hybrid 0.5 5000000 0.0001 64
test_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt 5000000

gen: 
	bash gen.sh
train_all_rnn:
	bash train.sh rnn 2 5000000 0.0003 320
	bash train.sh rnn 1 5000000 0.0003 512
	bash train.sh rnn 0.5 5000000 0.0003 1024
eval_all_rnn:
	bash eval.sh rnn 0.5 model_best.pt 5000000
	bash eval.sh rnn 1 model_best.pt 5000000
	bash eval.sh rnn 2 model_best.pt 5000000
train_all_hybrid:
	bash train.sh hybrid 0.5 5000000 0.0001 512
	bash train.sh hybrid 1 5000000 0.0001 512
	bash train.sh hybrid 2 5000000 0.0001 512
eval_all_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt 5000000
	bash eval.sh hybrid 1 model_best.pt 5000000
	bash eval.sh hybrid 2 model_best.pt 5000000
train_main_transformer:
	bash train.sh transformer 2 5000000 0.0001 512
	bash train.sh transformer 1 5000000 0.0001 512
	bash train.sh transformer 0.5 5000000 0.0001 512
eval_main_transformer:
	bash eval.sh transformer 2 model_best.pt 5000000
	bash eval.sh transformer 1 model_best.pt 5000000
	bash eval.sh transformer 0.5 model_best.pt 5000000
train_all_transformer_1:
	bash train.sh transformer 0.125 5000000 0.0001 1800
	bash train.sh transformer 0.5 5000000 0.0001 1800
	bash train.sh transformer 1 5000000 0.0001 1800
	bash train.sh transformer 2 5000000 0.0001 1800
	bash train.sh transformer 3 5000000 0.0001 1800
eval_all_transformer_1:
	bash eval.sh transformer 0.125 model_best.pt 5000000
	bash eval.sh transformer 1 model_best.pt 5000000
	bash eval.sh transformer 2 model_best.pt 5000000
	bash eval.sh transformer 3 model_best.pt 5000000
train_all_transformer_2:
	bash train.sh transformer 5 5000000 0.0001 1800
	bash train.sh transformer 7 5000000 0.0001 1800
	bash train.sh transformer 9 5000000 0.0001 1800
	bash train.sh transformer 0.25 5000000 0.0001 1800
eval_all_transformer_2:
	bash eval.sh transformer 5 model_best.pt 5000000
	bash eval.sh transformer 7 model_best.pt 5000000
	bash eval.sh transformer 9 model_best.pt 5000000
	bash eval.sh transformer 0.25 model_best.pt 5000000
train_all_transformer: train_all_transformer_1 train_all_transformer_2
eval_all_transformer: eval_all_transformer_1 eval_all_transformer_2
