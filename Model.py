import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from collections import OrderedDict

class PMFN(nn.Module):
    def __init__(self, label_dim=1):
        super(PMFN, self).__init__()
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=64)
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)

        fusion_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(nn.Linear(32, label_dim))

    def forward(self, img, omics):
        img_ = torch.sum(img, dim=2, keepdim=False)  # [1, 1, 256]
        omic_ = omics.unsqueeze(dim=1)  # [1, 1, 32]

        for i in range(3):
            if i == 0:
                out_img = self.path_transformer(img_)  # [1, 1, 256]
                out_omic = self.omic_transformer(omic_)  # [1, 1, 32]
                attention_scores = torch.matmul(out_omic.permute(0, 2, 1), out_img)  # [1, 32, 256]
                attention_weights = F.softmax(attention_scores, dim=-1)  # [1, 32, 256]
                out_fused = torch.matmul(attention_weights, out_img.permute(0, 2, 1))  # [64, 32, 1]
            else:
                out_img = self.fusion_transformer(out_fused.permute(0, 2, 1))
                out_omic = self.omic_transformer(omic_)
                attention_scores = torch.matmul(out_omic.permute(0, 2, 1), out_img)  # [1, 32, 256]
                attention_weights = F.softmax(attention_scores, dim=-1)  # [1, 32, 256]
                out_fused = torch.matmul(attention_weights, out_img.permute(0, 2, 1))

        prediction = self.classifier(out_fused.view(out_fused.shape[0], -1))

        return prediction


class SNN_omics(nn.Module):
    def __init__(self, input_dim=320, omic_dim=32, dropout_rate=0.25, act=None, label_dim=1):
        super(SNN_omics, self).__init__()
        self.input_dim = input_dim
        self.omic_dim = omic_dim
        hidden = [64, 48, 32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], self.omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(self.omic_dim, label_dim))

    def forward(self, omic):
        features = self.encoder(omic)  # [1, 32]
        # features = self.classifier(features)

        return features


class FC_slide(nn.Module):
    def __init__(self, input_dim=256, dropout_rate=0, label_dim=1):
        super(FC_slide, self).__init__()
        self.input_dim = input_dim

        hidden = [256, 128, 64, 32]

        encoder1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], hidden[3]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(hidden[3], label_dim))

    def forward(self, img):
        img_ = torch.sum(img, dim=2, keepdim=False)  # [1, 1, 256]
        features = self.encoder(img_)
        out = self.classifier(features)

        return out