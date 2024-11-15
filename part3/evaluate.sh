python3 rnn/val.py \
    --model_type rnn \
    --dataset_dir ./data/rnn/cot_binary_32_1000000_right  \
    --output_dir ./output_rnn \
    --model_dir ./output_rnn/model_1.pt\
    --model_config_path ./configs/0.5m_rnn.json \
    --batch_size 1
