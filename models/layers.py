import torch
import torch.nn as nn
from torch.nn import Parameter
import math
class forget_1(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []

        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]
            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                f_t = torch.sigmoid(ax_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)
                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)
                # output gate
                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)
                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t
                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq

class forget_2(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        # GCN系数
        self.Wn1 = Parameter(torch.Tensor(p, p))
        self.bn1 = Parameter(torch.Tensor(p))
        self.Wn2 = Parameter(torch.Tensor(p, p))
        self.bn2 = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        # math.sqrt 平方根
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):

        """
        assumes x.shape represents (字段长度/时间步总数, batch_size, 输入x的特征数)
        """

        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []

        # 如果没给定h_t、c_t
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                # x变两维
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                # GCN公式
                ax_t = a_t @ x_t @ self.Wn1 + self.bn1
                ax_t = ax_t @ self.Wn2 + self.bn2

                # input gate
                # sigmoid，范围在（0，1）
                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                # +
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                # print(i_t[1, :5, :5])

                # forget gate
                f_t = torch.sigmoid(ax_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                # cell 记忆池
                # tanh，范围在（-1，1）
                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                # output gate
                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)

                # t时刻真正的记忆状态
                c_t = f_t @ c + i_t @ g_t
                # 更新c
                c = c_t

                # 真正的模型输出
                h_t = o_t @ torch.tanh(c_t)
                # 更新h
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))

        # 在第0维将其结合
        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq


class forget_3(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        # GCN系数
        self.Wn1 = Parameter(torch.Tensor(p, p))
        self.bn1 = Parameter(torch.Tensor(p))
        self.Wn2 = Parameter(torch.Tensor(p, p))
        self.bn2 = Parameter(torch.Tensor(p))
        self.Wn3 = Parameter(torch.Tensor(p, p))
        self.bn3 = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell 记忆池
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                ax_t = a_t @ x_t @ self.Wn1 + self.bn1
                ax_t = ax_t @ self.Wn2 + self.bn2
                ax_t = ax_t @ self.Wn3 + self.bn3
                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                # +
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                f_t = torch.sigmoid(ax_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)
                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                # output gate
                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)

                c_t = f_t @ c + i_t @ g_t

                c = c_t

                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq

class input_1(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]

                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(ax_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)
                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq
class input_2(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]

                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(ax_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)
                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq
class input_3(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.W = Parameter(torch.Tensor(hidden_size, 1))
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]

                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(ax_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)
                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq



class output_1(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W = Parameter(torch.Tensor(hidden_size, 1))
        # GCN系数
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                # +
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                # print(i_t[1, :5, :5])

                # forget gate
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                # output gate
                o_t = torch.sigmoid(ax_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)

                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq

class output_2(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W = Parameter(torch.Tensor(hidden_size, 1))
        # GCN系数
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                # +
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                # print(i_t[1, :5, :5])

                # forget gate
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                # output gate
                o_t = torch.sigmoid(ax_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)

                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq


class output_3(nn.Module):
    def __init__(self, data_row=30, p=10, hidden_size=100, num_layers=1):
        super().__init__()
        self.data_row = data_row
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W = Parameter(torch.Tensor(hidden_size, 1))
        # GCN系数
        self.Wn = Parameter(torch.Tensor(p, p))
        self.bn = Parameter(torch.Tensor(p))

        # input gate
        self.W_ii = Parameter(torch.Tensor(p, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(torch.Tensor(hidden_size))
        self.b_hi = Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.W_if = Parameter(torch.Tensor(p, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(torch.Tensor(hidden_size))
        self.b_hf = Parameter(torch.Tensor(hidden_size))

        # cell
        self.W_ig = Parameter(torch.Tensor(p, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(torch.Tensor(hidden_size))
        self.b_hg = Parameter(torch.Tensor(hidden_size))

        # output gate
        self.W_io = Parameter(torch.Tensor(p, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(torch.Tensor(hidden_size))
        self.b_ho = Parameter(torch.Tensor(hidden_size))

        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        seq_len, batch_size, _, _ = x[0].size()
        hidden_seq = []

        h_seq = []
        c_seq = []
        h0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)
        c0 = torch.randn(self.num_layers, batch_size, self.data_row, self.hidden_size)

        for i in range(self.num_layers):
            h = h0[i, :, :, :]
            c = c0[i, :, :, :]

            for t in range(seq_len):
                x_t = x[0][t, :, :, :]
                a_t = x[1][t, :, :, :]
                ax_t = a_t @ x_t @ self.Wn + self.bn

                i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)

                # +
                i_t = (i_t @ self.W).squeeze()
                i_t = torch.diag_embed(i_t)
                # print(i_t[1, :5, :5])

                # forget gate
                f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)

                # +
                f_t = (f_t @ self.W).squeeze()
                f_t = torch.diag_embed(f_t)

                g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)

                # output gate
                o_t = torch.sigmoid(ax_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)

                # +
                o_t = (o_t @ self.W).squeeze()
                o_t = torch.diag_embed(o_t)

                c_t = f_t @ c + i_t @ g_t
                c = c_t
                h_t = o_t @ torch.tanh(c_t)
                h = h_t

                hidden_seq.append(h_t.unsqueeze(0))

            h_seq.append(h_t.unsqueeze(0))
            c_seq.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).reshape(seq_len, batch_size, -1)

        return hidden_seq