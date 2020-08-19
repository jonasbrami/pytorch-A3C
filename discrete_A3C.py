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
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
KERNEL_SIZE = 3 



env = retro.make('SonicTheHedgehog-Sms')
N_S = env.observation_space.shape[2]
N_A = env.action_space.n
env.close()

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        
        # self.extract_features_net = nn.Sequential(nn.Conv2d(N_S, 32, KERNEL_SIZE, stride=2, padding=1),#torch.Size([1, 3, 192, 256])
        #                                  nn.ReLU(),#torch.Size([1, 32, 96, 128])
        #                                  nn.Conv2d(32, 32, KERNEL_SIZE, stride=2, padding=1),
        #                                  nn.ReLU(),#torch.Size([1, 32, 48, 64])
        #                                  nn.Conv2d(32, 32, KERNEL_SIZE, stride=2, padding=1),
        #                                  nn.ReLU(),#torch.Size([1, 32, 24, 32])
        #                                  nn.Conv2d(32, 32, KERNEL_SIZE, stride=2, padding=1),
        #                                  nn.ReLU()) #torch.Size([1, 32, 12, 16])
                                         
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        self.conv1 = nn.Conv2d(3 , 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.actor = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, a_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 1))
        

        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTMCell(32 * 12 * 16 , 256)
        self.hx , self.cx = (None, None)

    def forward(self, x):
        #extracted_features = self.extract_features_net(x)
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))
        x = nn.ReLU(self.conv4(x))
        (self.hx, self.cx) = self.lstm( x, (self.hx, self.cx) )
        print('tata',self.hx.shape)
        return self.actor(self.hx), self.critic(self.hx)

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
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
        self.env = retro.make('SonicTheHedgehog-Sms')

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                s = np.moveaxis(s,-1,0) #channel first
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    gnet = Net(N_S, N_A)         # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
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
