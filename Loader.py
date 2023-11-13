import PIL
import os
import torch
from torch.utils import data
import pandas as pd
from util import *
import numpy as np
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, dataframe, split=None, transfer=None):
        self.dataframe = dataframe
        if split != None:
            index_split = self.dataframe[dataframe['Split'] == split].index
            self.dataframe = self.dataframe.loc[index_split, :]
        self.transfer = transfer
        self.length = len(self.dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        patch = self.dataframe.iloc[item, :]
        survival_label = patch[['days', 'event']].values.astype('float')
        # TCGA-GBMLGG
        if patch['Grade'] == 'G2':
            subtype_label = 0
        elif patch['Grade'] == 'G3':
            subtype_label = 1
        else:
            subtype_label = 2
        # TCGA-LUSC
        # if patch['Grade'] == 'N0':
        #     subtype_label = 0
        # elif patch['Grade'] == 'N1':
        #     subtype_label = 1
        # elif patch['Grade'] == 'N2':
        #     subtype_label = 2
        # else:
        #     subtype_label = 3
        # TCGA-BRCA
        # subtype_label = patch['Grade']

        slide_path = "./data/gbmlgg/slide/entropy_v_s/de_class/" + patch['img']  # slide Loader
        slide = np.load(slide_path)  # [1, 64, 256]
        slide = np.transpose(slide, (1, 2, 0))
        if self.transfer != None:
            slide = self.transfer(slide)

        omics = patch.drop(["PatientID", "img", "Cluster", "Split", "days", "event", "Grade"]).values
        omics = torch.FloatTensor(omics.astype(float))

        # print(omics.shape, slide.shape, survival_label, subtype_label)

        return omics, slide, survival_label, subtype_label, patch['PatientID']


    def split_cluster(file, split, cluster, transfer=None):
        df = pd.read_csv(file)
        index = df[df['Cluster'] == cluster].index

        return PatchData(df.loc[index, :], split=split, transfer=transfer)
