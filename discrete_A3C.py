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


UPDATE_GLOBAL_ITER = 50
GAMMA = 0.9
MAX_EP = 100000
KERNEL_SIZE = 3
NUM_WORKERS = 4
ENV = 'Breakout-v0' 
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

        self.extract_features_net = nn.Sequential(nn.Conv2d(N_S, 16, KERNEL_SIZE, stride=2, padding=1),#torch.Size([1, 3, 192, 256])
                                         nn.ReLU(),#torch.Size([1, 32, 96, 128])
                                         nn.Conv2d(16, 32, KERNEL_SIZE, stride=2, padding=1),
                                         nn.ReLU(),#torch.Size([1, 32, 48, 64])
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(32, 64, 2, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2)
        )
        
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor = nn.Sequential(#nn.Linear(256, 128),
                                   #nn.Tanh(),
                                   nn.Linear(64, a_dim))
        self.critic = nn.Sequential(#nn.Linear(256, 128),
                                    #nn.Tanh(),
                                    nn.Linear(64, 1))

        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTMCell(32 * 5 * 7, 128)#256)

    def forward(self, x):

        x = x.cuda()
        x = self.extract_features_net(x)
        x = x.view(-1, 64)
        
        return self.actor(x), self.critic(x)

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=-1).data
        return prob.multinomial(num_samples=1).detach()


    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
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
        if self.name == 'w00':
            b_loss = []
        else:
            b_loss = None
        while self.g_ep.value < MAX_EP:
            total_step = 1
            s = env.reset()
            s = cv2.resize(s, IMG_SIZE1)
            s = cv2.resize(s, IMG_SIZE2)
            s = np.moveaxis(s, -1, 0)  # channel first
            #s = np.zeros((192, 256, 3))

            buffer_s, buffer_a, buffer_r= [], [], []
            ep_r = 0.

            time0 = time.time()
            while True:
                if self.name == 'w00':
                    env.render()
                a = self.lnet.choose_action(
                    v_wrap(s[None, :]))
               

                actionArray = np.zeros(N_A, dtype=bool)
                actionArray[a] = 1
                
                s_, r, done, _ = env.step(actionArray)
                s_ = cv2.resize(s_, IMG_SIZE1)
                s_ = cv2.resize(s_, IMG_SIZE2)

                s_ = np.moveaxis(s_, -1, 0)  # channel first
                
                done = done or total_step == MAX_EPI_LENGHT

                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)


                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_,
                                  buffer_s, buffer_a, buffer_r, GAMMA, b_loss)
                    buffer_s, buffer_a, buffer_r = [
                    ], [], []

                    if done:
                        if self.name == 'w00':
                            print('loss ' + str(b_loss[-1]))
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
