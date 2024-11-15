import torch
import torch.nn as nn
from torch.nn import functional as F

import attention as attention

torch.manual_seed(0)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    rope = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
class MultiHeadAttention(nn.Module):
    def __init__(self, config, p_drop=0.1):
        super().__init__()
        d_model = config.n_embd
        num_heads = config.n_head
        # GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size, n_layer=4, n_head=8, n_embd=256)
        assert d_model % num_heads == 0, 'num_heads must divide d_model'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn = attention.CausalSelfAttention(config)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        x = self.attn(x)
        x = self.drop(self.proj(x))
        return x
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
class MLP(nn.Module):
    def __init__(self, d_model, p_drop):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2*d_model)
        self.act1 = nn.GELU()   
        self.fc2 = nn.Linear(2*d_model, 4*d_model)
        #mini model
        #self.fc2 = nn.Linear(2*d_model, 3*d_model)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(4*d_model, 2*d_model)
        #self.fc3 = nn.Linear(3*d_model, 2*d_model)
        self.act3 = nn.GELU()
        self.fc4 = nn.Linear(2*d_model, d_model)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        x=self.act2(x)
        x=self.fc3(x)
        x=self.act3(x)
        x=self.fc4(x)
        x=self.drop(x)
        return x
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        ### TODO: Implement the Block class
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Residual(MultiHeadAttention(config))
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = Residual(MLP(config.n_embd,p_drop=config.resid_pdrop))

    def forward(self, x):
        ### TODO: Implement the forward pass, you can try post-norm or pre-norm here
        x = self.attn(x)
        x = self.ln1(x)
        x = self.mlp(x)
        x = self.ln2(x)
        return x

class GPT(nn.Module):

    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.rope = config.rope
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        # TODO: Implement the forward pass
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        if not self.rope:
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
        else:
            x = self.drop(token_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        # TODOEND

        return logits, loss
