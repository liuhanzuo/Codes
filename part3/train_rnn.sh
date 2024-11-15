for i in $(seq 0 $1); do
    echo "Training iteration $i"
    python3 rnn/train.py \
        --model_type rnn \
        --dataset_dir ./data/rnn/cot_binary_32_1000000_right \
        --output_dir ./output_rnn \
        --previous_model_path ./output_rnn/model_$(($i+$2)).pt \
        --model_path ./output_rnn/model_$(($i+1+$2)).pt \
        --model_config_path ./configs/0.5m_rnn.json \
        --batch_size 128 \
        --total_training_samples 500000 \
        --lr 0.0003
done