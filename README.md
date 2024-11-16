# Codes

what you should do to run the code:

```
cd part3
pip install -r dependecy.txt
```

also make sure you have conda environment and can run bash&cuda

```
make gen
```

to generate data, this may take a while for not longer 10 minutes, the data will be stored in ``./part3/data``

also, since we will use the pretrained model from huggingface , you may need to use clash(however if you are in America, it may not be necessary)

```
cd clash
./clash -d .
```

then train the rnn,hybri model by run

```
make train_all_rnn
make train_all_hybrid
make train_all_transformer_1
make train_all_transformer_2
```

evaluate them by

```
make eval_all_rnn
make eval_all_hybrid
make eval_all_transformer_1
make eval_all_transformer_2
```

since the evaluation will test the performace of CoT, it may take a while.

**Note that I will run transformer, since they are easier to run, so please help me to run rnn and hybrid first!**

***Thanks a lot***
