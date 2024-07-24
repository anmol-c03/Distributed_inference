import torch
from torch import nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
import inspect


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
        # self.dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # if self.flash:
        #     # efficient attention using Flash Attention CUDA kernels
        #     'a really faster way to implement self attention is flash attention'
        #     y = F.scaled_dot_product_attention(q, k, v,is_causal=True)
        # else:
            # manual implementation of attention
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # wei = self.attn_dropout(wei)
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
        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        # self.out=self.dropout(x)
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

        #weights tying concept
        # self.transformer.wte.weight=self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'set_scale'):
                std=(2*self.config.n_layer )**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx,layer_num,target=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        # if layer_num==0:
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            loss=None

        return logits,loss 

    @classmethod
    def from_pretrained(cls, model_type,override_args=None,):
        # assert model_num is not None
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        print(sd_keys)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # model.load_state_dict(sd_hf)
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def config_optimizer(self,weight_decay,lr,betas,device):
        params={pn:p for pn,p in self.named_parameters()}
        params={pn:p for pn,p in params.items() if p.requires_grad}
        decayed_points=[p for pn,p in params.items() if p.dim()>=2]
        non_decayed_points=[p for pn,p in params.items() if p.dim()<2]
        optimizer_groups=[
            {'params':decayed_points,'weight_decay':weight_decay},
            {'params':non_decayed_points,'weight_decay':0}]
        num_decay_params=sum(p.numel() for p in decayed_points)
        num_nodecay_params=sum(p.numel() for p in non_decayed_points)
        print(f'there are {len(decayed_points)} decayed_params with total num of parameters = {num_decay_params}')
        print(f'there are {len(non_decayed_points)} non_decayed_params with total num of parameters = {num_nodecay_params}')
        # Create AdamW optimizer and use the fused version if it is available
        #if cuda available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()


        optimizer=torch.optim.AdamW(optimizer_groups,lr=lr,betas=betas,eps=1e-8,fused=use_fused)
        return optimizer
    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond=idx[:,-self.config.block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            prob=F.softmax(logits,dim=-1)
            topk_prob,topk_index=torch.topk(prob,50,dim=-1)
            next_token=torch.multinomial(topk_prob,num_samples=1)  
            x_col=torch.gather(topk_index,-1,next_token)
            idx=torch.cat((idx,x_col),dim=1)
        return idx 
    

# torch.manual_seed(1335)

# model=GPT.from_pretrained('gpt2')
# # conf=GPTConfig()
# # model=GPT(conf)

# import tiktoken
# print('form model.generate\n')
# enc=tiktoken.get_encoding('gpt2')
# x=enc.encode("Hello, I'm a language model")
# tokens=torch.tensor(x,dtype=torch.long)
# x=tokens.unsqueeze(0)
# x=x.repeat(5,1)
# print('------->',x.shape)
# output=model.generate(x,20)

# for i in range(5):
#     print(enc.decode(output[i].tolist()))



# print('from .ipynb')
# import torch.nn.functional as F

# while x.size(1) < 30:
#     with torch.no_grad():
#         logits=model(x)[0]  # returns logits,loss [0] for loss only
#         logits=logits[:, -1, :]
#         probs=F.softmax(logits,dim=-1)
#         topk_probs,topk_indices=torch.topk(probs,50,dim=-1)
#         next_token=torch.multinomial(topk_probs, num_samples=1)
#         xcol=torch.gather(topk_indices,-1,next_token)
#         x=torch.cat((x, xcol), dim=1)

# for i in range(5):
#     tokens=x[i,:]
#     text=enc.decode(tokens.tolist())
#     print('->',text)

# print('from original gpt2')
# from transformers import GPT2LMHeadModel

# model=GPT2LMHeadModel.from_pretrained('gpt2')
# from transformers import pipeline,set_seed

# set_seed(42)
# pipe=pipeline('text-generation',model='gpt2',max_length=20)   

# print(pipe("Hello, I'm a language model",num_return_sequences=5))