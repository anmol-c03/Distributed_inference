import torch
from torch import nn
import math
import torch.nn.functional as F
from dataclasses import dataclass



class  CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config. n_embd)
        self.c_proj.set_scale=1.0
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size), persistent=False)

        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        y = wei @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        self.out=self.c_proj(y)
        return self.out

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.set_scale=1.0

    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
        
        
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size :int=1024
    vocab_size :int=50257
    n_layer :int=1
    n_head :int=12
    n_embd :int=768
    bias :bool=True

class GPT(nn.Module):
    def __init__(self,config,layer_num,iter) -> None:
        super().__init__()
        self.config=config

        if layer_num==0:
            self.transformer=nn.ModuleDict(dict(
                    wte=nn.Embedding(config.vocab_size,config.n_embd),
                    wpe=nn.Embedding(config.block_size,config.n_embd),
                    # drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    ln_f=nn.LayerNorm(config.n_embd)
            ))
        else:
            self.transformer=nn.ModuleDict(dict(
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    ln_f=nn.LayerNorm(config.n_embd)
            ))

        if layer_num==iter-1:
            self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)


    def forward(self, idx,layer_num,iter,target=None):
        # forward the GPT model itself
        if layer_num==0:
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = tok_emb + pos_emb
        else:
            x=idx
        for block in self.transformer.h:
            x = block(x)
        if layer_num==iter-1:
            x = self.transformer.ln_f(x)
            x = self.lm_head(x) 
        logits=x
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            loss=None

        return logits,loss 


