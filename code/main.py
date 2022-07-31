import numpy as np
import scipy.sparse as sp
import random
import gc

from clac_metric import get_metrics
from utils import constructHNet, constructNet, get_edge_index,load_data, Sizes
import torch as t
from torch import optim
from loss import Myloss
from case_study import *
import MKGAT


def train(model, train_data, optimizer, sizes):
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(train_data['Y_train'], score, model.mirna_l, model.disease_l, model.alpha1,
                               model.alpha2, sizes)
        model.alpha1 = t.mm(
            t.mm((t.mm(model.mirna_k, model.mirna_k) + model.lambda1 * model.mirna_l).inverse(), model.mirna_k),
            2 * train_data['Y_train'] - t.mm(model.alpha2.T, model.disease_k.T)).detach()
        model.alpha2 = t.mm(t.mm((t.mm(model.disease_k, model.disease_k) + model.lambda2 * model.disease_l).inverse(), model.disease_k),
                            2 * train_data['Y_train'].T - t.mm(model.alpha1.T, model.mirna_k.T)).detach()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(1, sizes.epoch + 1):
        train_reg_loss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass


def PredictScore(train_mirna_disease_matrix, mirna_matrix, disease_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['Y_train'] = t.DoubleTensor(train_mirna_disease_matrix)
    feature = constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix)
    feature = t.FloatTensor(feature)
    train_data['feature'] = feature


    adj = constructNet(train_mirna_disease_matrix)
    adj = t.FloatTensor(adj)
    adj_edge_index = get_edge_index(adj)
    train_data['Adj'] = {'data': adj, 'edge_index': adj_edge_index}

    model = MKGAT.Model(sizes, mirna_matrix, disease_matrix)
    print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())

    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate)

    train(model, train_data, optimizer, sizes)
    return model(train_data)






if __name__ == "__main__":
    data_path = '../data/dateset/'
    data_set = 'MDA2.0/'
    


    mirna_sim = np.loadtxt(data_path + data_set + 'sm.csv', delimiter=',')
    disease_sim = np.loadtxt(data_path + data_set + 'sd.csv')
    mirna_disease = np.loadtxt(data_path + data_association + 'association.csv', delimiter=',')
    sizes = Sizes(mirna_disease.shape[0], mirna_disease.shape[1])

