import numpy as np
import torch as t


def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, mirna_disease_matrix))
    mat2 = np.hstack((mirna_disease_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    return np.vstack((mat1, mat2))


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = t.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = t.where(t.isinf(D_5), t.full_like(D_5, 0), D_5)
    L_D_11 = t.mm(D_5, L_D_1)
    L_D_11 = t.mm(L_D_11, D_5)
    return L_D_11


def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = t.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne


def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[t.nonzero(k, as_tuple=False)[0]]
    K[t.where(K == 0)] = min_v
    D = t.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S

def load_data(directory):
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    D_SSM = (D_SSM1 + D_SSM2) / 2

    ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))
    IM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if D_SSM[i][j] == 0:
                ID[i][j] = D_GSM[i][j]
            else:
                ID[i][j] = D_SSM[i][j]
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_FSM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
            else:
                IM[i][j] = M_FSM[i][j]
    return ID, IM

class Sizes(object):
    def __init__(self, mira_size, disease_size):
        self.mira_size = mira_size
        self.disease_size = disease_size
        self.F1 = 128
        self.F2 = 64
        self.F3 = 32
        self.k_fold = 5
        self.epoch = 10
        self.learn_rate = 0.001
        self.seed = 1
        self.h1_gamma = 2 ** (-5)
        self.h2_gamma = 2 ** (-5)
        self.h3_gamma = 2 ** (-5)


        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-3.7)
