import torch
import torch.nn as nn
import numpy as np

use_cuda = torch.cuda.is_available()


def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)  # 用于返回一个矩阵主对角线以下的下三角矩阵，其它元素全部为0

    return indicator_matrix

class NegativeLogLikelihood(nn.Module):
    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        '''Calculate the average Cox negative partial log-likelihood.
                Input:
                    pred: linear predictors from trained model.
                    ytime: true survival time from load_data().
                    yevent: true censoring status from load_data().
                Output:
                    cost: the cost that is to be minimized.
        '''
        ytime, yevent = label[:, 0], label[:, 1]
        n_observed = yevent.sum(0)
        if use_cuda:
            ytime_indicator = R_set(ytime).cuda()
        else:
            ytime_indicator = R_set(ytime)

        risk_set_sum = ytime_indicator.mm(torch.exp(pred))
        diff = pred - torch.log(risk_set_sum)  # + 1e-10
        yevent = yevent.view(len(pred), -1).to(dtype=torch.float)
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        if n_observed == 0:
            cost = torch.tensor(0.0, requires_grad=True).view(1, ).cuda()
        else:
            cost = (- (sum_diff_in_observed / n_observed)).view(1, )

        return cost, n_observed

def CoxLoss(label, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    survtime, censor = label[:, 0], label[:, 1]
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def RankingLoss(label, hazard_pred, device):
    survtime, censor = label[:, 0], label[:, 1]
    R = hazard_pred.reshape(-1)
    N = len(survtime)
    T = (N*(N-1))/2
    loss_rank = 0
    for i in range(N):
        for j in range(N):
            theta = torch.sigmoid(R[i] - R[j])
            exp_theta = torch.exp(-1 * theta)
            if survtime[i] > survtime[j]:
                loss = torch.log(1 + exp_theta)
            if survtime[i] == survtime[j]:
                loss = 0.5 * theta + torch.log(1 + exp_theta)
            else:
                loss = theta + torch.log(1 + exp_theta)
            loss_rank += loss
    return loss_rank/T
