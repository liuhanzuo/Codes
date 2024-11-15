for i in $(seq 0 $1); do
    echo "Training iteration $i"
    python3 rnn/train.py \
        --model_type transformer \
        --dataset_dir ./data/rnn/cot_binary_32_1000000_right \
        --output_dir ./output_transformer \
        --previous_model_path ./output_transformer/model_$(($i+$2)).pt \
        --model_path ./output_transformer/model_$(($i+1+$2)).pt \
        --model_config_path ./configs/0.5m_transformer.json \
        --batch_size 64 \
        --total_training_samples 5000000 \
        --lr 0.0001
done