echo "model_type: $1"
echo "dataset_dir: ./data/rnn/cot_binary_32_1000000_right"
echo "output_dir: ./output_$1_$2_$4"
echo "model_dir: ./output_$1_$2_$4/$3"
echo "model_config_path: ./configs/$2m_transformer.json"
echo "batch_size: 1"
python3 rnn/val.py \
    --model_type $1 \
    --dataset_dir ./data/rnn/cot_binary_32_1000000_right  \
    --output_dir ./output_$1_$2_$4 \
    --model_dir ./output_$1_$2_$4/$3\
    --model_config_path ./configs/$2m_transformer.json \
    --batch_size 1
