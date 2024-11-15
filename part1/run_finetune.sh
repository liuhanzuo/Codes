

# finetune with/without pretrained

python run.py \
    --function finetune \
    --outputs_path ./output/finetune_with_pretrained.pt\
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --finetune_corpus_path ./dataset/finetune/birth_places_train.tsv \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path ./output/pretrain.pt