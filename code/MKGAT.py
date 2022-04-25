import torch as t
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import conv
from utils import *
import numpy as np

class Model(nn.Module):
    def __init__(self, sizes, mirna_sim, disease_sim):
        super(Model, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.mira_size = sizes.mira_size
        self.disease_size = sizes.disease_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3
        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma

        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 4

        self.att_d = Parameter(t.ones((1, 4)), requires_grad=True)
        self.att_m = Parameter(t.ones((1, 4)), requires_grad=True)


        self.mirna_sim = t.DoubleTensor(mirna_sim)
        self.disease_sim = t.DoubleTensor(disease_sim)

        self.gcn_1 = conv.GATConv(self.mira_size + self.disease_size, self.F1)
        self.gcn_2 = conv.GATConv(self.F1, self.F2)
        self.gcn_3 = conv.GATConv(self.F2, self.F3)



        self.alpha1 = t.randn(self.mira_size, self.disease_size).double()
        self.alpha2 = t.randn(self.disease_size, self.mira_size).double()

        self.mirna_l = []
        self.disease_l = []

        self.mirna_k = []
        self.disease_k = []

    def forward(self, input):
        t.manual_seed(self.seed)
        x = input['feature']
        adj = input['Adj']
        mirna_kernels = []
        disease_kernels = []

        H1 = t.relu(self.gcn_1(x, adj['edge_index']))
        mirna_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.mira_size].clone(), 0, self.h1_gamma, True).double()))
        disease_kernels.append(t.DoubleTensor(getGipKernel(H1[self.mira_size:].clone(), 0, self.h1_gamma, True).double()))


        H2 = t.relu(self.gcn_2(H1, adj['edge_index']))
        mirna_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.mira_size].clone(), 0, self.h2_gamma, True).double()))
        disease_kernels.append(t.DoubleTensor(getGipKernel(H2[self.mira_size:].clone(), 0, self.h2_gamma, True).double()))


        H3 = t.relu(self.gcn_3(H2, adj['edge_index']))
        mirna_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.mira_size].clone(), 0, self.h3_gamma, True).double()))
        disease_kernels.append(t.DoubleTensor(getGipKernel(H3[self.mira_size:].clone(), 0, self.h3_gamma, True).double()))

        mirna_kernels.append(self.mirna_sim)
        disease_kernels.append(self.disease_sim)

        mirna_k = sum([self.att_m[0][i] * mirna_kernels[i] for i in range(self.kernel_len)])
        self.mirna_k = normalized_kernel(mirna_k)
        disease_k = sum([self.att_d[0][i] * disease_kernels[i] for i in range(self.kernel_len)])
        self.disease_k = normalized_kernel(disease_k)
        self.mirna_l = laplacian(mirna_k)
        self.disease_l = laplacian(disease_k)

        out1 = t.mm(self.mirna_k, self.alpha1)
        out2 = t.mm(self.disease_k, self.alpha2)

        out = (out1 + out2.T) / 2

        return out
