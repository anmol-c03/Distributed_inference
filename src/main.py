import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import model_top

model_base=GPT2LMHeadModel.from_pretrained('gpt2')
sd_base=model_base.state_dict()
# import sys;sys.exit(0)


def get_layers_weights(layer):
    return model_base.transformer.h[layer].state_dict()


models={}
'''
model=model_top.GPT(model_top.GPTConfig())
# print(model.state_dict().keys())
sd=model.state_dict()
sd_keys=sd.keys()
sd_keys = [k for k in sd_keys if not k.endswith('.ln_f.weight')] 
sd_keys = [k for k in sd_keys if not k.endswith('.ln_f.bias')] 
list_sd_keys=list(sd_keys)
# print(sd_keys)
import sys;sys.exit(0)
'''


# sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
# print('\n\n')

# print(model_hf.keys())
# import sys;sys.exit(0)

iter=model_base.config.n_layer
for i in range(iter):
    model=model_top.GPT(model_top.GPTConfig(),i,iter)
    # print(model.state_dict().keys())
    # break
    sd=model.state_dict()
    sd_keys=sd.keys()
    list_sd_keys=list(sd_keys)

    
    sd_hf=get_layers_weights(i)
    sd_keys_hf = sd_hf.keys()


   
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    list_sd_keys_hf=list(sd_keys_hf)

    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    if i==0:
        token,pos='transformer.wte.weight','transformer.wpe.weight'
        sd[token].copy_(sd_base[token])
        sd[pos].copy_(sd_base[pos])
        j=2
    else: 
        j=0
    for k in range(len(list_sd_keys_hf)):
        key,key_hf=list_sd_keys[j],list_sd_keys_hf[k]
        # print(key,key_hf)
        if any(key_hf.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[key_hf].shape[::-1] == sd[key].shape
            with torch.no_grad():
                sd[key].copy_(sd_hf[key_hf].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[key_hf].shape == sd[key].shape
            with torch.no_grad():
                sd[key].copy_(sd_hf[key_hf])
        j+=1
    sd['transformer.ln_f.weight'].copy_(sd_base['transformer.ln_f.weight'])
    sd['transformer.ln_f.bias'].copy_(sd_base['transformer.ln_f.bias'])
    if i==iter-1:
        sd['lm_head.weight'].copy_(sd_base['lm_head.weight'])   

    models[i]=model

for i in range(iter):
    print(f'Model {i}:')
    print(models[i].state_dict().keys())
    print('\n')