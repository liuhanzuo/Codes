echo "model_type: $1"
echo "dataset_dir: ./data/rnn/cot_binary_32_1000000_right"
echo "output_dir: ./output_$1_$2_$3"
echo "model_config_path: ./configs/$2m_transformer.json"
echo "batch_size: $5"
echo "total_training_samples: $3"
echo "lr: $4"
python3 rnn/train.py \
    --model_type $1 \
    --dataset_dir ./data/rnn/no_cot_binary_32_1000000_right \
    --output_dir ./output_$1_$2_$3 \
    --model_config_path ./configs/$2m_transformer.json \
    --batch_size $5 \
    --total_training_samples $3 \
    --lr $4