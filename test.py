import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lifelines.utils import concordance_index as con_index

from Model import SNN_omics, PMFN, FC_slide
import Loader

from integrated_gradients import integrated_gradients
import pandas as pd
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_test = 64
# ----------------------

transfers = transforms.Compose([
    transforms.ToTensor()
])

result = []
# Cluster is set to the number of cross-validation
cluster = 0
valid0 = Loader.PatchData.split_cluster('./data/gbmlgg/omics/TCGA-GBMLGG-omic.csv', 'Train', cluster, transfer=transfers)
dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=True, num_workers=0)

omics_model = SNN_omics(input_dim=80, omic_dim=32)
omics_model.to(device)
model = PMFN()
model.to(device)
omics_model.load_state_dict(torch.load('./model/omics_model.pt'))
model.load_state_dict(torch.load('./model/model.pt'))

# Validation
model.eval()
omics_model.eval()
c_index = 0
accuracy = 0
n = 0
risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
for iteration, data in enumerate(dataloader_var):
    n += 1.0
    # with torch.no_grad():
    omics, img_feature, s_label, c_label, sample_id = data
    img_var = Variable(img_feature, requires_grad=False).cuda()
    s_label_var = Variable(s_label, requires_grad=False).cuda()
    c_label_var = Variable(c_label, requires_grad=False).cuda()
    omics_var = Variable(omics, requires_grad=True).cuda()
    ytime, yevent = s_label_var[:, 0], s_label_var[:, 1]
    y, e = ytime.detach().cpu().numpy(), yevent.detach().cpu().numpy()

    # pred_test = omics_model(omics_var)
    omics_feature_var = omics_model(omics_var)
    pred_test = model(img_var, omics_feature_var)

    # Integral gradient algorithm
    # ig_attr = integrated_gradients(omics_model, omics_var.detach().cpu().numpy(), target=1, n_iter=1000, device='cuda',visualize=False)
    #
    # for i in range(len(sample_id)):
    #     name = sample_id[i].split()
    #     data = ig_attr[i, :].tolist()
    #     name_data = name + [y[i]] + [e[i]] + data
    #     f = open('./results/brca_omics_ig_attr.csv', "a", newline="")
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(name_data)
    #     f.close()

    pred_test = pred_test.reshape(-1)
    risk_pred = pred_test.detach().cpu().numpy()

    # Saving the prediction output for KM curve plotting
    # for i in range(len(sample_id)):
    #     name = sample_id[i].split()
    #     name_data = name + [y[i]] + [e[i]] + [omics[i][0].detach().cpu().numpy()] + [omics[i][1].detach().cpu().numpy()] + [risk_pred[i]]
    #     f = open('./results/brca_pred_omics_1.csv', "a", newline="")
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(name_data)
    #     f.close()

    risk_pred_all = np.concatenate((risk_pred_all, risk_pred.reshape(-1)))
    censor_all = np.concatenate((censor_all, e.reshape(-1)))
    survtime_all = np.concatenate((survtime_all, y.reshape(-1)))
    try:
        c_index = con_index(survtime_all, -risk_pred_all, censor_all)
        print('({}):{}'.format(n, c_index))
    except:
        print('No admissable pairs in the dataset.')
    result.append(c_index)
print('max: {}, min: {}, mean c-index: {}'.format(max(result), min(result), sum(result)/len(result)))
