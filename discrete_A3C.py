import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import retro
import numpy as np
import cv2
import time
os.environ["OMP_NUM_THREADS"] = "12"
os.environ['LANG'] = 'en_US'


def use_gpu(x=True): return torch.set_default_tensor_type(torch.cuda.FloatTensor
                                                          if torch.cuda.is_available() and x
                                                          else torch.FloatTensor)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

use_gpu()


UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 5000
KERNEL_SIZE = 3
NUM_WORKERS = 4
ENV = 'PongDeterministic-v4'
#'PongDeterministic-v4'
# 'SonicTheHedgehog-Sms'

env = gym.make(ENV)
res = env.observation_space.shape
N_S = env.observation_space.shape[2]
N_A = env.action_space.n
env.close()

MAX_EPI_LENGHT = 3000
IMG_SIZE1 = ((int)(res[0]/2), (int)(res[1]/2))
IMG_SIZE2 = ((int)(res[0]/4), (int)(res[1]/4))


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.extract_features_net = nn.Sequential(nn.Conv2d(N_S, 32, KERNEL_SIZE, stride=2, padding=1),#torch.Size([1, 3, 192, 256])
                                         nn.ReLU(),#torch.Size([1, 32, 96, 128])
                                         nn.Conv2d(32, 32, KERNEL_SIZE, stride=2, padding=1),
                                         nn.ReLU(),#torch.Size([1, 32, 48, 64])
                                         nn.Conv2d(32, 32, KERNEL_SIZE, stride=2, padding=1),
                                         nn.ReLU()#torch.Size([1, 32, 24, 32])
        )
        
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.actor = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, a_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 1))

        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTMCell(32 * 5 * 7, 256)

    def forward(self, x, lstm_hx_cx):
        (hxs, cxs) = lstm_hx_cx

        x = x.cuda()
        x = self.extract_features_net(x)
        x = x.view(-1, 32 * 5 * 7)
        if hxs:
            if len(x) == 1:
                (hx, cx) = self.lstm(
                    x, (hxs[-1], cxs[-1]))
            else:
                (hx, cx) = self.lstm(
                    x, (torch.cat(hxs[-(len(x) + 1): -1], dim=0), torch.cat(cxs[-(len(x) + 1): -1], dim=0)))

        else:
            (hx, cx) = self.lstm(x)

        return self.actor(hx), self.critic(hx), (hx, cx)

    def choose_action(self, s, lstm_hx_cx):
        self.eval()
        logits, _, (hx, cx) = self.forward(s, lstm_hx_cx)
        prob = F.softmax(logits, dim=-1).data
        #m = self.distribution(prob)

        return prob.multinomial(num_samples=1).detach(), (hx, cx)

        # return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t, lstm_hx_cx):
        self.train()
        logits, values, _ = self.forward(s, lstm_hx_cx)
        td = v_t.cuda() - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a.cuda()) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        env = gym.make(ENV)

        while self.g_ep.value < MAX_EP:
            total_step = 1
            s = env.reset()
            s = cv2.resize(s, IMG_SIZE1)
            s = cv2.resize(s, IMG_SIZE2)
            s = np.moveaxis(s, -1, 0)  # channel first
            #s = np.zeros((192, 256, 3))

            buffer_s, buffer_a, buffer_r, buffer_hx, buffer_cx = [], [], [], [], []
            ep_r = 0.

            time0 = time.time()
            while True:
                # if self.name == 'w00':
                #     env.render()
                a, (hx, cx) = self.lnet.choose_action(
                    v_wrap(s[None, :]), (buffer_hx, buffer_cx))
                if not buffer_hx:
                    buffer_hx.append(torch.zeros_like(hx))
                    buffer_cx.append(torch.zeros_like(cx))

                actionArray = np.zeros(N_A, dtype=bool)
                actionArray[a] = 1
                
                s_, r, done, _ = env.step(actionArray)
                s_ = cv2.resize(s_, IMG_SIZE1)
                s_ = cv2.resize(s_, IMG_SIZE2)

                s_ = np.moveaxis(s_, -1, 0)  # channel first
                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                buffer_hx.append(hx)
                buffer_cx.append(cx)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done or total_step == MAX_EPI_LENGHT:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_,
                                  buffer_s, buffer_a, buffer_r, GAMMA, (buffer_hx, buffer_cx))
                    buffer_s, buffer_a, buffer_r, buffer_hx, buffer_cx = [
                    ], [], [], [buffer_hx[-1].detach().clone()], [buffer_cx[-1].detach().clone()]

                    if done or total_step == MAX_EPI_LENGHT:
                        record(self.g_ep, self.g_ep_r, ep_r,
                               self.res_queue, self.name)
                        print("Epoch duration: " +
                              str(time.time()-time0) + " seconds")
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)
        env.close()


if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    # torch.autograd.set_detect_anomaly(True)
    mp.set_start_method("spawn")
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters())  # , lr=1e-4,
    #  betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i)
               for i in range(NUM_WORKERS)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
