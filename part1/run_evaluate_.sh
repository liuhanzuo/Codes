python run.py \
    --function evaluate \
    --outputs_path ./output/predictions_mini.txt \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path ./output/finetune_with_pretrained_mini.pt