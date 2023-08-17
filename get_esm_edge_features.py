import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm
from torch.utils.data import DataLoader, Dataset

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# path
Dataset_Path = './Data/'
Model_Path = './Model/'
Result_Path = './Result/'


class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['gene'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['solubility'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]

        return sequence_name, sequence, label

    def __len__(self):
        return len(self.labels)


def get_edge_fe(model, dataloader, batch_converter):
    model.cuda()
    model.eval()
    for data in tqdm(dataloader):
        # print(data)
        sequence_names, sequence, labels = data
        if len(sequence[0]) > 1022:
            print(sequence_names[0])
        elif len(sequence[0]) <= 1022:
            data1 = [(sequence_names[0], sequence[0])]
            batch_labels, batch_strs, batch_tokens = batch_converter(data1)
            if len(batch_tokens) <= 1024:
                with torch.no_grad():
                    results = model(batch_tokens.cuda(), repr_layers=[36],
                                    return_contacts=True)
                contacts = results["contacts"]

                np.save('Data/features/new_edge/' + sequence_names[0] + '.npy',
                        np.array(torch.squeeze(contacts).cpu()))  # 第一个是数据存放位置


if __name__ == "__main__":
    # # eSol数据集
    train_dataframe = pd.read_csv('Data/eSol_train.csv', sep=',')
    test_dataframe = pd.read_csv(Dataset_Path + 'eSol_test.csv', sep=',')

    # #s数据集
    s_dataframe = pd.read_csv('Data/scerevisiae/scerevisiae_test.csv', sep=',')
    # all_dataframe = pd.concat([train_dataframe, test_dataframe, s_dataframe], axis=0)
    all_dataframe = test_dataframe.loc[test_dataframe['gene'] == 'tyrS']

    dataset = ProDataset(all_dataframe)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    print("dataloader initialised")

    # pre_trained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    pre_trained_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    print(1)
    batch_converter = alphabet.get_batch_converter()
    print(2)
    pre_trained_model.eval()
    # atten_infer = pre_trained_model.cuda()
    print("pretrained model loaded and ")

    # get_node_fe(model=pre_trained_model, dataloader=dataloader, batch_converter=batch_converter)
    get_edge_fe(model=pre_trained_model, dataloader=dataloader, batch_converter=batch_converter)
