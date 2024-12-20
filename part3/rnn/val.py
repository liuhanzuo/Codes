from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP 
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.nn as nn
import numpy as np
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import os
import json

from utils import set_seed, parse_args
from data import load_dataset, IsTreeDataset
from model import get_transformer_model, get_rnn_model, get_hybrid_model

@torch.no_grad()
def evaluate(model, val_loader, args, log_file=None):
    model.eval()
    total_loss = 0
    total_correct_samples = 0
    total_samples = 0
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        for batch in tqdm(val_loader, total=len(val_loader)):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            if args.model_type == 'transformer':
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(input_ids)[0]
                loss = criterion(logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1))
                
            logits = logits[..., :-1, :].argmax(dim=-1)
            total_loss += loss.item()
            mask = labels[..., 1:] != -100
            if log_file is not None:
                file = open(log_file, 'a')
                for i in range(logits.size(0)):
                    file.write(f'input: {input_ids[i]}\n')
                    file.write(f'output: {logits[i]}\n')
                    file.write(f'label: {labels[i]}\n')
                    file.write(f'mask: {mask[i]}\n')
                    file.write('\n')
            sample_acc = (logits == labels[..., 1:]) | ~mask  # True for correct predictions and ignored tokens
            sample_acc = sample_acc.all(dim=-1)  # Check if all tokens in a sample are correct/ignored
            total_correct_samples += sample_acc.sum().item()
            total_samples += labels.size(0)  # Count each sample in the batch
    return total_loss / len(val_loader), total_correct_samples / total_samples

def generate(model, input_ids, max_len, args):
    model.eval()
    input_ids = input_ids.to(args.device)
    outputs = input_ids
    with torch.no_grad():
        for i in range(max_len):
            logits = model(outputs)[0]
            logits = logits[:, -1, :]
            next_token = logits.argmax()
            outputs = torch.cat([outputs, next_token.reshape(1,1)], dim=1)
    outputs = outputs[:, input_ids.size(1):]
    return outputs

@torch.no_grad()
def evaluate_through_generation(model, val_loader, val_dataset, args):
    model.eval()
    total_correct_samples = 0
    total_samples = 0
    for sample in tqdm(val_loader, total=len(val_loader)):
        input_ids = sample['input_ids']
        input_ids = input_ids.to(args.device)
        # find the first appearence of [EOS] token, which is the end of the input sequence
        eos_indices = input_ids.eq(val_dataset.vocab['[EOS]']).nonzero()
        if eos_indices.size(0) > 0:
            eos_index = eos_indices[0, 1].item()
            real_input_ids = input_ids[:, :eos_index+1]
        else:
            real_input_ids = input_ids

        if args.model_type == 'transformer':
            generated = model.generate(real_input_ids, max_length=input_ids.size(1) + 1, do_sample=False, num_return_sequences=1, eos_token_id=val_dataset.vocab['[TRUE]'], pad_token_id=val_dataset.vocab['[PAD]'])
        else:
            generated = generate(model, real_input_ids, input_ids.size(1) + 1 - real_input_ids.size(1), args)
        # from IPython import embed; embed()
        # print(f'generated: {val_dataset.convert_ids_to_tokens(generated[0])}')
        # find the first appearance of the [TRUE] and [FALSE] token, check if it's the same as the last token of input_ids
        true_indices = generated.eq(val_dataset.vocab['[TRUE]']).nonzero() # true_indices means the index of [TRUE] token in the generated sequence
        false_indices = generated.eq(val_dataset.vocab['[FALSE]']).nonzero() # false_indices means the index of [FALSE] token in the generated sequence
        true_index = true_indices[0, 1].item() if true_indices.size(0) > 0 else -1
        false_index = false_indices[0, 1].item() if false_indices.size(0) > 0 else -1
        # print(true_index, false_index)
        result = val_dataset.vocab['[TRUE]']
        with open(os.path.join(args.output_dir, 'generation_log.txt'), 'a') as f:
            f.write(f'Generated: {val_dataset.convert_ids_to_tokens(generated[0])}\n')
            f.write(f'Ground Truth: {val_dataset.convert_ids_to_tokens(input_ids[0])}\n')
            f.write('\n')
        if false_index != -1 and (true_index == -1 or true_index > false_index):
            result = val_dataset.vocab['[FALSE]']
        if false_index == -1 and true_index == -1:
            result = -1
        if result == input_ids[0, -1].item():
            total_correct_samples += 1
        
        total_samples += 1

        #print(f'Accuracy: {total_correct_samples / total_samples}')
    print(f'Accuracy: {total_correct_samples / total_samples}')
    return total_correct_samples / total_samples


def main():
    args = parse_args()
    print(args.output_dir)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.report_to_wandb:
        wandb.init()
    set_seed(args.seed)
    val_dataset = load_dataset(os.path.join(args.dataset_dir, 'val'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    if args.model_type == 'transformer':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_transformer_model(
                val_dataset,
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                max_position_embeddings=config['max_position_embeddings']
            )
        else:
            model = get_transformer_model(
                val_dataset,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=20,
                num_attention_heads=8,
                max_position_embeddings=4096
            )
    elif args.model_type == 'rnn':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_rnn_model(
                val_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers']
            )
        else:
            model = get_rnn_model(
                val_dataset,
                hidden_size = 128,
                num_hidden_layers= 10
            )
    elif args.model_type == 'hybrid':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_hybrid_model(
                val_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                max_position_embeddings=config['max_position_embeddings'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size']
            )
        else:
            model = get_hybrid_model(
                val_dataset,
                hidden_size=128,
                num_hidden_layers=9
            )
    model = model.to(device=args.device, dtype=torch.float32)
    print(model)
    model.load_state_dict(torch.load(args.model_dir))
    #model = model.from_pretrained(args.model_dir)
    if args.model_type == 'rnn':
        model.lm_head.weight = model.backbone.embedding.weight
    if args.model_type == 'transformer':
        model.lm_head.weight = model.model.embed_tokens.weight
    model = model.to(device=args.device, dtype=torch.float32)
    log_file = args.output_dir + '/log.txt' if args.output_dir is not None else None
    val_loss , val_acc = evaluate(model, val_loader, args, log_file)
    val_acc_cot = evaluate_through_generation(model, val_loader, val_dataset, args)
    print(f'Initial | val loss: {val_loss} | val acc: {val_acc}')
    print(f'COT | val acc: {val_acc_cot}')
    json.dump({
        'val_acc': val_acc,
        'val_loss': val_loss,
        'val_acc_cot': val_acc_cot,
    }, open(f'{args.output_dir}/eval_results.json', 'w'))

if __name__ == '__main__':
    main()