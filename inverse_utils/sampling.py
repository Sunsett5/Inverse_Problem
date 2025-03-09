import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def uncond_sampling(x, seq, model, b, cls_fn=None, classes=None):
    # torch.cuda.empty_cache()
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    eta = 1.0
    with torch.no_grad():
        xt = x
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            if et.size(1) == 6:
                et = et[:, :3]
            c1 = eta * ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
            c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # x0_t = x0_t.clip(-1, 1)
            xt_next = at_next.sqrt() * x0_t + et * c2 + c1 * torch.randn_like(x0_t)
            xt = xt_next
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds