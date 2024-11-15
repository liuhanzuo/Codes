python3 rnn/val.py \
    --model_type $1 \
    --dataset_dir ./data/rnn/cot_binary_32_1000000_right  \
    --output_dir ./output_$1 \
    --model_dir ./output_$1/$2\
    --model_config_path ./configs/$3 \
    --batch_size 1
