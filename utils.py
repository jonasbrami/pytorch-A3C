"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import time
from os import path

use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor 
                                             if torch.cuda.is_available() and x 
                                             else torch.FloatTensor)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

use_gpu()

CHECK_PATH = "./checkpoints/model.pt"

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, lstm_hx_cx):
    if done:
        v_s_ = 0.               # terminal
    else:
        _, v, _ = lnet.forward(v_wrap(s_[None, :]), lstm_hx_cx)
        v_s_ = v.data[0,0] #.cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    
    loss = lnet.loss_func(
        v_wrap(np.array(bs)),
        # v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(
        #     np.vstack(ba)),
        torch.stack(ba),
        v_wrap(np.array(buffer_v_target)[:, None]),
        lstm_hx_cx)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()

    # t0 = time.time()
    loss.backward(retain_graph=True)
    # print('{} seconds'.format(time.time() - t0))

    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())



def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % ep_r#global_ep_r.value,
    )


def save_checkpoint(model, opt, epoch):
    torch.save({   
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'opt_state_dict': opt.state_dict()
                }, CHECK_PATH)

def load_checkpoint(model, opt, global_ep):
    if path.exists(CHECK_PATH):
        checkpoint = torch.load(CHECK_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['opt_state_dict'])
        global_ep.value = checkpoint['epoch']
