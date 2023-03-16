import torch

import numpy as np

def auc_link_prediction(A, A_predict, n):
    n1 = 0
    n2 = 0
    # print(np.nonzero(A))
    [x, y] = np.nonzero(A)
    # print(x, y)
    x2 = np.argwhere(A == 0)
    # print(x2)
    for i in range(n):
        index1 = np.random.randint(0, len(x), 1)
        # print(1, index1)
        index2 = np.random.randint(0, len(x2), 1)
        # print(2, index2)
        temp1 = A_predict[x[index1], y[index1]]
        temp2 = A_predict[x2[index2, 0], x2[index2, 1]]
        if temp1 > temp2:
            n1 += 1
        if temp1 == temp2:
            n2 += 1

    return (n1 + 0.5 * n2) / n

def precision_link_prediction(A, a_predict, L):
    n = 0
    a_predict = np.reshape(a_predict, (-1))
    A = np.reshape(A, (-1))
    a_predict_argsort = (-a_predict).argsort()
    # print(a_predict_argsort)
    for i in range(L):
        # print(A[a_predict_argsort[i]])
        if A[a_predict_argsort[i]] == 1:
            n += 1
    precision = n / L
    return precision
def gcn_a(A, symmetric=True):
    d = A.sum(1)
    if symmetric:
        D = torch.diag_embed(torch.pow(d, -0.5))
        return D.matmul(A).matmul(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.matmul(A)
def gcn_x(X, p):
    b1 = X.sum(1).unsqueeze(-1)
    i = 1
    a = X
    while i <= (p - 1):
        a = a.matmul(X)
        b = a.sum(1).unsqueeze(-1)
        if i == 1:
            A_X = torch.cat([b1, b], dim=2)
        else:
            A_X = torch.cat([A_X, b], dim=2)
        i += 1
    return A_X