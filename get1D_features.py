import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm
from torch.utils.data import DataLoader, Dataset

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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


def get_matrix(model, dataloader, batch_converter):
    model.cuda()
    model.eval()
    for data in tqdm(dataloader):
        # print(data)
        sequence_names, sequence, labels = data
        if len(sequence[0]) > 1015:
            print(sequence_names[0])
        elif len(sequence[0]) <= 1015:
            data1 = [(sequence_names[0], sequence[0])]
            batch_labels, batch_strs, batch_tokens = batch_converter(data1)
            if len(batch_tokens) <= 1024:
                with torch.no_grad():
                    results = model(batch_tokens.cuda(), repr_layers=[33],
                                    return_contacts=False)
                attention = results["representations"][33]
                np.save('./Data/features/node_features2/' + sequence_names[0] + '.npy',
                        np.array(torch.squeeze(attention).cpu()))   # 第一个是数据存放位置


def cal_mean_std():
    total_length = 0
    mean = np.zeros(1371)
    mean_square = np.zeros(1371)
    for name in tqdm(os.listdir('./Data/features/node_features/')):
        matrix = np.load('./Data/features/node_features2/' + name)
        # matrix_two = np.load('./Data/node_features/' + name)
        matrix_two = np.load('./Data/node_features/' + name)
        matrix = matrix[1:matrix.shape[0] - 1:]
        matrix = np.concatenate([matrix, matrix_two], axis=1)
        total_length += matrix.shape[0]
        mean += np.sum(matrix, axis=0)
        mean_square += np.sum(np.square(matrix), axis=0)

    mean /= total_length  # EX
    mean_square /= total_length  # E(X^2)
    std = np.sqrt(np.subtract(mean_square, np.square(mean)))  # DX = E(X^2)-(EX)^2, std = sqrt(DX)

    np.save('./Data/eSol_2_1371_mean.npy', mean)
    np.save('./Data/eSol_2_1371_std.npy', std)


def csv_to_npy():
    # train_dataframe = pd.read_csv(Dataset_Path + 'eSol_train.csv', sep=',')
    # test_dataframe = pd.read_csv(Dataset_Path + 'eSol_test.csv', sep=',')
    # dataframe = pd.concat([train_dataframe, test_dataframe], axis=0)
    for name in tqdm(os.listdir('./Data/node_features_21/')):
        seq_name = name[:-4]
        seq = pd.read_csv('./Data/fasta/' + seq_name)
        seq = seq.iat[0, 0]
        matrix = read_spot_single('./Data/node_features_21/' + name, seq)
        np.save('./Data/node_features_23/' + seq_name + '.npy', matrix)



rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"
ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
           185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))


def angle_norm(angle):
    rad_angle = np.deg2rad(angle)
    angle_split = (np.concatenate([np.sin(rad_angle), np.cos(rad_angle)], 1) + 1) / 2.
    return angle_split


def read_spot_single(file_name, seq):
    data = pd.read_csv(file_name)
    ss3_prob = np.concatenate(
        (data['P3C'].to_numpy()[:, None], data['P3E'].to_numpy()[:, None], data['P3H'].to_numpy()[:, None]), 1).astype(
        np.float32)
    ss8_prob = np.concatenate((
        data['P8C'].to_numpy()[:, None], data['P8S'].to_numpy()[:, None], data['P8T'].to_numpy()[:, None],
        data['P8H'].to_numpy()[:, None], data['P8G'].to_numpy()[:, None],
        data['P8I'].to_numpy()[:, None], data['P8E'].to_numpy()[:, None], data['P8B'].to_numpy()[:, None]), 1).astype(
        np.float32)

    ASA_den = np.array([dict_rnam1_ASA[i] for i in seq]).astype(np.float32)[:, None]
    asa = data['ASA'].to_numpy()[:, None]
    asa_relative = np.clip(asa / ASA_den, 0, 1)

    hseu = data['HseU'].to_numpy()[:, None]
    hsed = data['HseD'].to_numpy()[:, None]
    CN = data['CN'].to_numpy()[:, None]

    psi = data['Psi'].to_numpy()[:, None]
    psi_split = angle_norm(psi)
    phi = data['Phi'].to_numpy()[:, None]
    phi_split = angle_norm(phi)
    theta = data['Theta'].to_numpy()[:, None]
    theta_split = angle_norm(theta)
    tau = data['Tau'].to_numpy()[:, None]
    tau_split = angle_norm(tau)
    spot_single_feat = np.concatenate(
        (ss3_prob, ss8_prob, asa_relative, hseu, hsed, CN, psi_split, phi_split, theta_split, tau_split), 1)
    return spot_single_feat


if __name__ == "__main__":
    # # get_matrix()
    # cal_mean_std()
    # csv_to_npy()
    train_dataframe = pd.read_csv(Dataset_Path + 'eSol_train.csv', sep=',')
    test_dataframe = pd.read_csv(Dataset_Path + 'eSol_test.csv', sep=',')
    dataframe = pd.concat([train_dataframe, test_dataframe], axis=0)
    dataset = ProDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    print("dataloader initialised")

    ### obj creation for pre-trained model
    # pre_trained_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    # batch_converter = alphabet.get_batch_converter()
    pre_trained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    pre_trained_model.eval()
    # atten_infer = pre_trained_model.cuda()
    print("pretrained model loaded and ")

    get_matrix(model=pre_trained_model, dataloader=dataloader, batch_converter=batch_converter)


