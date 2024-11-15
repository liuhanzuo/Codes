python run.py \
    --function finetune \
    --outputs_path ./output/finetune_without_pretrained.pt\
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --finetune_corpus_path ./dataset/finetune/birth_places_train.tsv \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \

python run.py \
    --function evaluate \
    --outputs_path ./output/predictions_withoutpretrain.txt \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path ./output/finetune_without_pretrained.pt