import torch
import numpy as np
import random

@torch.no_grad()
def save(method, dataset, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, './models/%s/%s.pt'%(method, dataset))
    with open('./models/%s/%s.log'%(method, dataset), 'w') as f:
        f.write(str(ep)+' '+str(acc))


def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset))
    model.load_state_dict(st, False)
    return model

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False