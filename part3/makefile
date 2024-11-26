train_transformer:
	bash train.sh transformer 1 2000000 0.0001 320
test_transformer:
	bash eval.sh transformer 0.5 model_best.pt 2000000
train_and_test_transformer: train_transformer test_transformer
	
train_rnn:
	bash train_test.sh rnn 0.5 5000000 0.003 256
test_rnn:
	bash eval.sh rnn 0.5 model_best.pt 2000000
train_and_test_rnn: train_rnn test_rnn

train_hybrid:
	bash train.sh hybrid 0.5 2000000 0.001 256
test_hybrid:
	bash eval.sh hybrid 1 model_best.pt 2000d000
love:
	bash train.sh hybrid 1 2000000 0.002 256
	bash train.sh hybrid 2 2000000 0.002 256
	bash train.sh rnn 2 2000000 0.003 256
	bash train.sh rnn 1 2000000 0.003 256
	bash train.sh rnn 0.5 2000000 0.003 256
gen: 
	bash gen.sh
train_all_rnn:
	bash train.sh rnn 2 2000000 0.003 256
	bash train.sh rnn 1 2000000 0.003 256
	bash train.sh rnn 0.5 2000000 0.003 256
eval_all_rnn:
	bash eval.sh rnn 0.5 model_best.pt 2000000
	bash eval.sh rnn 1 model_best.pt 2000000
	bash eval.sh rnn 2 model_best.pt 2000000
train_all_hybrid:  
	bash train.sh hybrid 2 2000000 0.001 320
	bash train.sh hybrid 1 2000000 0.001 320
	bash train.sh hybrid 0.5 2000000 0.001 320
eval_all_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt 2000000
	bash eval.sh hybrid 1 model_best.pt 2000000
	bash eval.sh hybrid 2 model_best.pt 2000000
train_main_transformer:
	bash train_trans.sh transformer 0.5 2500000 0.0003 128
	bash train_trans.sh transformer 1 2500000 0.0003 128
	bash train_trans.sh transformer 2 2500000 0.0003 128
eval_main_transformer:
	bash eval.sh transformer 2 model_best.pt 2000000
	bash eval.sh transformer 1 model_best.pt 2000000
	bash eval.sh transformer 0.5 model_best.pt 2000000
train_all_transformer_1:
	bash train.sh transformer 0.125 2000000 0.0001 320
	bash train.sh transformer 0.5 2000000 0.0001 320
	bash train.sh transformer 1 2000000 0.0001 320
	bash train.sh transformer 2 2000000 0.0001 320
	bash train.sh transformer 3 2000000 0.0001 320
eval_all_transformer_1:
	bash eval.sh transformer 0.125 model_best.pt 2000000
	bash eval.sh transformer 1 model_best.pt 2000000
	bash eval.sh transformer 2 model_best.pt 2000000
	bash eval.sh transformer 3 model_best.pt 2000000
train_all_transformer_2:
	bash train.sh transformer 5 2000000 0.002 320
	bash train.sh transformer 7 2000000 0.002 320
	bash train.sh transformer 9 2000000 0.002 320
	bash train.sh transformer 0.25 2000000 0.002 320
eval_all_transformer_2:
	bash eval.sh transformer 5 model_best.pt 2000000
	bash eval.sh transformer 7 model_best.pt 2000000
	bash eval.sh transformer 9 model_best.pt 2000000
	bash eval.sh transformer 0.25 model_best.pt 2000000
train_all_transformer: train_all_transformer_1 train_all_transformer_2
eval_all_transformer: eval_all_transformer_1 eval_all_transformer_2
