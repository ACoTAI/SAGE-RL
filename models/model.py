from models.utils import gcn_a,gcn_x

from models.layers import forget_1
from models.layers import forget_2
from models.layers import forget_3
from models.layers import input_1
from models.layers import input_2
from models.layers import input_3
from models.layers import output_1
from models.layers import output_2
from models.layers import output_3

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.Data import Data
import matplotlib.pyplot as plt

p = 15
hidden_size = 1000
num_layers = 1
# -----------------------ia-contacts----------------------------
data_row = 113
# data_row = 30
input_size = data_row * data_row
output_size = data_row * data_row
data_name = '../data/ia_contacts.csv'
data_name = Data().load_data(data_name)
batch_index, train_x, train_y = Data().get_net_train_data(data_name)
# print(batch_index)

LR = 0.0002 


# self.gate: "input", self.gcn: 1
class NET0(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET0, self).__init__()
        self.rnn = input_1(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)
    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x

class NET1(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET1, self).__init__()
        self.rnn = forget_1(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x


# self.gate: "output", self.gcn: 1
class NET2(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET2, self).__init__()
        self.rnn = output_1(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x


# self.gate: "input", self.gcn: 1
class NET3(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET3, self).__init__()
        self.rnn = input_2(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)
    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x

class NET4(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET4, self).__init__()
        self.rnn = forget_2(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x


# self.gate: "output", self.gcn: 1
class NET5(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET5, self).__init__()
        self.rnn = output_2(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x

class NET6(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET6, self).__init__()
        self.rnn = input_3(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)
    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x

class NET7(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET7, self).__init__()
        self.rnn = forget_3(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x


# self.gate: "output", self.gcn: 1
class NET8(nn.Module):
    def __init__(self, data_row=data_row, p=p, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers):
        super(NET8, self).__init__()
        self.rnn = output_3(data_row, p, hidden_size, num_layers)
        self.reg1 = nn.Linear(data_row * hidden_size, 1000)
        self.reg2 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[-1, :, :]
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        return x



NET = []
net0 = NET0()
net1 = NET1()
net2 = NET2()
net3 = NET3()
net4 = NET4()
net5 = NET5()
net6 = NET6()
net7 = NET7()
net8 = NET8()
NET.extend([net0, net1, net2, net3, net4, net5, net6, net7, net8])

# 定义损失函数
LOSS = []
loss_func0 = nn.MSELoss()
loss_func1 = nn.MSELoss()
loss_func2 = nn.MSELoss()
loss_func3 = nn.MSELoss()
loss_func4 = nn.MSELoss()
loss_func5 = nn.MSELoss()
loss_func6 = nn.MSELoss()
loss_func7 = nn.MSELoss()
loss_func8 = nn.MSELoss()
LOSS.extend([loss_func0, loss_func1, loss_func2, loss_func3,
             loss_func4, loss_func5, loss_func6, loss_func7, loss_func8])
# print(LOSS)

# 定义优化器
OPT = []
optimizer0 = torch.optim.Adam(net0.parameters(), lr=LR)
optimizer1 = torch.optim.Adam(net1.parameters(), lr=LR)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=LR)
optimizer3 = torch.optim.Adam(net3.parameters(), lr=LR)
optimizer4 = torch.optim.Adam(net4.parameters(), lr=LR)
optimizer5 = torch.optim.Adam(net5.parameters(), lr=LR)
optimizer6 = torch.optim.Adam(net6.parameters(), lr=LR)
optimizer7 = torch.optim.Adam(net7.parameters(), lr=LR)
optimizer8 = torch.optim.Adam(net8.parameters(), lr=LR)
OPT.extend([optimizer0, optimizer1, optimizer2, optimizer3, optimizer4,
            optimizer5, optimizer6, optimizer7, optimizer8])