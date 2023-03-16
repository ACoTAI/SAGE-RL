import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import RL.My_env as Env
import matplotlib.pyplot as plt

BATCH_SIZE = 15
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 20
MEMORY_CAPACITY = 50
env = Env.MyEnv()
N_ACTIONS = len(env.action_space)

# -----------------------ia-contacts----------------------------
data_row = 113

N_STATES = data_row * data_row
path = 'model/ia_net.pkl'

episode = 800
a_list = [x for x in range(len(env.action_space))]  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
len_train_x = 15

loss_list = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 


        self.fc1 = nn.Linear(N_STATES, 50)

        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(50, N_ACTIONS)

        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        actions_value = self.out(x)


        return actions_value


class SAGEGCN(object):
    def __init__(self):


        self.eval_net, self.target_net = Net(), Net()


        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = [0] * MEMORY_CAPACITY

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()


    def choose_action(self, x):
        b_s = torch.FloatTensor(x)  # torch.Size([15, 12769])

        if np.random.uniform() < EPSILON:

            actions_value = self.eval_net.forward(b_s)  # torch.Size([15, 9])

            actions_value = torch.sum(actions_value, 0)  # torch.Size([9])

            index = torch.max(actions_value, 0)[1].data.numpy()
            action = a_list[index]

        else:
            action = np.random.choice(a_list)

        print("action:", action)
        return action

    def store_transition(self, s, a, r, s_):
        transition = [0] * 4
        transition[0] = s
        transition[1] = a
        transition[2] = r
        transition[3] = s_
        transition_index = self.memory_counter % MEMORY_CAPACITY
        self.memory[transition_index] = transition

        self.memory_counter += 1
    def learn(self):

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # [33 21 ... 5]
        b_memory, b_s, b_a, b_r, b_s_ = [], [], [], [], []
        for i in sample_index:
            b_memory.append(self.memory[i])
            b_s.append(self.memory[i][0])
            b_a.append(self.memory[i][1])
            b_r.append(self.memory[i][2])
            b_s_.append(self.memory[i][3])

        b_s = torch.FloatTensor(b_s)  # torch.Size([15, 15, 12769])
        b_a = torch.LongTensor(b_a)  # tensor([0, 2, 2, ..., 6, 3]) BATCH_SIZE个
        # b_r list len: 15
        b_r = torch.FloatTensor(b_r).reshape(BATCH_SIZE, 1, 1)  # torch.Size([15, 1, 1])
        # tensor([0.9350, ..., 0.9893]) BATCH_SIZE

        b_s_ = torch.FloatTensor(b_s_)  # torch.Size([15, 15, 12769])

        q_eval = self.eval_net(b_s)  # torch.Size([15, 15, 9])
        # q_eval_gather = torch.FloatTensor(np.zeros((BATCH_SIZE, len_train_x, 1)))  # torch.Size([15, 15, 1])
        q_eval_gather = torch.FloatTensor(np.zeros((BATCH_SIZE, len_train_x, 1)))  # torch.Size([15, 15, 1])

        q_next = self.target_net(b_s_).detach()  # torch.Size([15, 15, 9])
        # q_next_max = torch.FloatTensor(np.zeros((BATCH_SIZE, len_train_x, 1)))  # torch.Size([15, 15, 1])
        q_next_max = torch.FloatTensor(np.zeros((BATCH_SIZE, len_train_x, 1)))  # torch.Size([15, 15, 1])
        for i in range(BATCH_SIZE):
            q_eval_gather[i, :, 0] = q_eval[i, :, b_a[i]]  # torch.Size([15, 15, 1])
            index_temp = torch.max(q_next[i].sum(dim=0), 0)[1].numpy()
            q_next_max[i, :, 0] = q_next[i, :, index_temp]

        q_target = b_r + GAMMA * q_next_max
        loss = self.loss_func(q_target, q_eval_gather)

        print("loss:", loss.item())
        loss_list.append(loss.detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


dqn = SAGEGCN()
r_list = []
count = 0
ep = []

print("\nCollecting experience...")
for i in range(episode):
    print('Episode: %s' % i)
    s = env.reset()
    episode_reward_sum = 0

    while True:
        a = dqn.choose_action(s)

        s_, r, done = env.step(a)
        print("reward:", r)
        dqn.store_transition(s, a, r, s_)

        episode_reward_sum += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            print("-" * 50)

        if done:

            break

        s = s_


plt.figure()
x = [x for x in range(len(loss_list))]
plt.plot(x, loss_list)
plt.show()
loss_list = np.array(loss_list)
np.save('../draw/loss_ia.npy', loss_list)
# plt.savefig('loss.png')

torch.save(dqn.eval_net.state_dict(), path)
