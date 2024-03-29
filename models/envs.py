
import numpy as np
import torch
from Data import Data
import models.layers as NET

from models.model import gcn_a,gcn_x

# 定义常量
p = 15
# -----------------------ia-contacts----------------------------
data_row = 113
# data_row = 30
data_name = '../data/ia_contacts.csv'
input_size = data_row * data_row
output_size = data_row * data_row

data = Data().load_data(data_name)
# 训练集 list
train_x, train_y = Data().get_train_data(data)  # len 65

# 测试集 list
test_x, test_y = Data().get_test_data(data)

class MyEnv:
    def __init__(self):
        self.index = 0
        self.done = False
        self.train_x, self.train_y = train_x, train_y

        self.original_state = self.train_x[0]
        self.action_space = [NET.NET0(), NET.NET1(), NET.NET2(),
                             NET.NET3(), NET.NET4(), NET.NET5(),
                             NET.NET6(), NET.NET7(), NET.NET8()]
        self.state = None

    def reset(self):
        self.index = 0
        self.state = self.original_state
        return self.state
    def step(self, action):
        # input_ (torch.Size([15, 1, 113, 15]), torch.Size([15, 1, 113, 113])); y torch.Size([1, 12769])
        tr_x, tr_y = process_data(self.train_x[self.index], self.train_y[self.index])
        net = self.action_space[action]
        net.load_state_dict(torch.load('ia_parameters/net' + str(action) + '.pkl'))

        pre_y = net(tr_x)

        reward = compute_reward(tr_y, pre_y)

        self.index = self.index + 1

        self.state = self.train_x[self.index]
        next_state = self.state

        if self.index == 64:
            self.done = True

        return next_state, reward, self.done


class MyEnvPred:
    def __init__(self):

        self.test_predict = []
        self.index = 0
        self.done = False
        self.test_x, self.test_y = test_x, test_y

        self.original_state = self.test_x[0]
        self.action_space = [NET.NET0(), NET.NET1(), NET.NET2(),
                             NET.NET3(), NET.NET4(), NET.NET5(),
                             NET.NET6(), NET.NET7(), NET.NET8()]
        self.state = None

    def reset(self):
        self.index = 0
        self.state = self.original_state
        return self.state

    def step(self, action):
        tr_x, tr_y = process_data(self.test_x[self.index], self.test_y[self.index])
        net = self.action_space[action]
        net.load_state_dict(torch.load('ia_parameters/net' + str(action) + '.pkl'))
        pre_y = net(tr_x)

        pre_y_np = pre_y.detach().numpy().reshape(-1)
        self.test_predict.extend(pre_y_np)

        self.index = self.index + 1
        next_state = None

        if self.index == 20:
            self.done = True
        else:
            self.state = self.test_x[self.index]
            next_state = self.state

        return next_state, self.done
