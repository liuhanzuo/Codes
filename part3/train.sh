echo "model_type: $1"
echo "dataset_dir:  ~/autodl-tmp/data/rnn/cot_binary_32_1000000_right "
echo "output_dir: ./output_$1_$2_$3"
echo "model_config_path: ./configs/$2m_$1.json"
echo "batch_size: $5"
echo "total_training_samples: $3"
echo "lr: $4"
python3 rnn/train.py \
    --model_type $1 \
    --dataset_dir ~/autodl-tmp/data/rnn/cot_binary_32_1000000_right \
    --output_dir ./output_$1_$2_$3_test \
    --previous_model_path ./output_$1_$2_$3_test/model_best.pt \
    --model_config_path ./configs/$2m_$1.json \
    --batch_size $5 \
    --total_training_samples $3 \
    --report_to_wandb \
    --log_interval 5000 \
    --lr $4