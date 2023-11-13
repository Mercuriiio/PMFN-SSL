import torch
import Loader
import itertools
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lifelines.utils import concordance_index as con_index
from torch.utils.tensorboard import SummaryWriter
from Model import SNN_omics, PMFN, FC_slide
from NegativeLogLikelihood import CoxLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_train = 64
bch_size_test = 64
epoch_size = 80
base_lr = 0.001
writer = SummaryWriter('./log')
# ----------------------

transfers = transforms.Compose([
    transforms.ToTensor()
])

result = []
kfold = 1  # In practice, please set to 5 or other
for cluster in range(kfold):
    print("************** SPLIT (%d/%d) **************" % (cluster+1, kfold))
    train0 = Loader.PatchData.split_cluster('./data/gbmlgg/omics/TCGA-GBMLGG-omic.csv', 'Train', cluster, transfer=transfers)
    valid0 = Loader.PatchData.split_cluster('./data/gbmlgg/omics/TCGA-GBMLGG-omic.csv', 'Valid', cluster, transfer=transfers)
    dataloader = DataLoader(train0, batch_size=bch_size_train, shuffle=True, num_workers=0)
    dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=True, num_workers=0)

    slide_model = FC_slide(input_dim=256)
    slide_model.to(device)
    slide_model.train()

    omics_model = SNN_omics(input_dim=80, omic_dim=32)
    omics_model.to(device)
    omics_model.train()

    model = PMFN()
    model.to(device)
    model.train()
    # optimizer = torch.optim.Adam(slide_model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    # optimizer = torch.optim.Adam(omics_model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.Adam(itertools.chain(omics_model.parameters(), model.parameters()),
                                 lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)

    for epoch in range(epoch_size):
        loss_board = 0
        for iteration, data in enumerate(dataloader):
            omics, img_feature, s_label, c_label, _ = data
            omics = Variable(omics, requires_grad=False).to(device)
            img_feature = Variable(img_feature, requires_grad=False).to(device)
            s_label = Variable(s_label, requires_grad=False).to(device)
            c_label = Variable(c_label, requires_grad=False).to(device)

            # pred = slide_model(img_feature)
            # pred = omics_model(omics)
            omics_feature = omics_model(omics)
            pred = model(img_feature, omics_feature)

            loss = CoxLoss(s_label, pred, 'cuda')
            print('Epoch: {}/({})'.format(iteration, epoch+1), 'Train_loss: %.4f' %(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_board += loss
        #writer.add_scalar("Train_loss", loss_board, epoch)

        # Validation
        model.eval()
        omics_model.eval()
        slide_model.eval()
        c_index = 0
        accuracy = 0
        n = 0
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        for iteration, data in enumerate(dataloader_var):
            n += 1.0
            with torch.no_grad():
                omics, img_feature, s_label, c_label, _ = data
                img_var = Variable(img_feature, requires_grad=False).to(device)
                s_label_var = Variable(s_label, requires_grad=False).to(device)
                c_label_var = Variable(c_label, requires_grad=False).to(device)
                omics_var = Variable(omics, requires_grad=False).to(device)
                ytime, yevent = s_label_var[:, 0], s_label_var[:, 1]

                # pred_test = slide_model(img_var)
                # pred_test = omics_model(omics_var)
                omics_feature_var = omics_model(omics_var)
                pred_test = model(img_var, omics_feature_var)

                pred_test = pred_test.reshape(-1)
            y, risk_pred, e = ytime.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), yevent.detach().cpu().numpy()
            # print(y, risk_pred)
            risk_pred_all = np.concatenate((risk_pred_all, risk_pred.reshape(-1)))
            censor_all = np.concatenate((censor_all, e.reshape(-1)))
            survtime_all = np.concatenate((survtime_all, y.reshape(-1)))
        try:
            c_index = con_index(survtime_all, -risk_pred_all, censor_all)
        except:
            print('No admissable pairs in the dataset.')
        print('Epoch(' + str(epoch + 1) + ')',  'Train_loss: %.4f' % (loss_board.item()), 'Test_acc: %.4f' % (c_index))
        #writer.add_scalar("Test_acc", c_index, epoch)

        result.append(c_index)

        # if epoch % 10 == 9:
        #     torch.save(omics_model.state_dict(), './model/omics_model_{}.pt'.format(epoch+1))
        #     torch.save(model.state_dict(), './model/model_{}.pt'.format(epoch+1))
    print(result)
    print('max: {}, min: {}, mean c-index: {}'.format(max(result), min(result), sum(result)/len(result)))
