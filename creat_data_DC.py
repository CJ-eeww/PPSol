from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset_Path = './Data/'
Model_Path = './Model/'
Result_Path = './Result/'

amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# 加载蛋白质名字，序列，标签
def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})

# 加载蛋白质特征
def load_features(sequence_name, sequence, mean, std):
    # len(sequence) * 20
    # blosum_matrix = np.array([blosum[i] for i in sequence])
    # len(sequence) * 71

    # 加载esm特征
    oneD_matrix = np.load(Dataset_Path + 'features/node_features2/' + sequence_name + '.npy')
    oneD_matrix = oneD_matrix[1:len(sequence) + 1:]
    twoD_matrix = np.load(Dataset_Path + 'node_features/' + sequence_name + '.npy')
    # 方差归一化
    # twoD_matrix = (twoD_matrix - mean) / std
    # 最大最小归一化

    twoD_matrix1 = (twoD_matrix - mean) / (std - mean)
    twoD_matrix = np.concatenate([twoD_matrix[:, :20], twoD_matrix1[:, 20:70], twoD_matrix[:, 70:]], axis=1)

    feature_matrix = np.concatenate([oneD_matrix, twoD_matrix], axis=1)

    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'features/new_edge/' + sequence_name + '.npy').astype(np.float32)
    matrix = matrix.reshape([matrix.shape[0], matrix.shape[1]])
    matrix1 = np.ones(matrix.shape, dtype=float)
    mask1 = np.tril(np.ones(matrix.shape, dtype=float), -2)
    mask2 = np.triu(np.ones(matrix.shape, dtype=float), 2)
    mask = mask1 + mask2
    matrix1 -= mask
    matrix = matrix + matrix1

    return matrix


def load_values():

    # 最大最下归一化
    mean = np.load(Dataset_Path + 'train_1_91_min.npy')
    std = np.load(Dataset_Path + 'train_1_91_max.npy')

    return mean, std

def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result


def load_gobal(name):

    global_feature = np.load(Dataset_Path + 'gobal_features/' + name + '.npy')
    mean = np.load(Dataset_Path + 'eSol_gobal_mean.npy')
    std = np.load(Dataset_Path + 'eSol_gobal_std.npy')

    return (global_feature - mean) / std


def protein_to_graph(protein_name, protein, p=None):
    mean, std = load_values()
    sequence_feature = load_features(protein_name, protein, mean, std)
    sequence_graph = load_graph(protein_name)
    if p is not None:
        sequence_graph = np.where(sequence_graph > p, 1, 0)
        sequence_graph = np.argwhere(sequence_graph == 1)
        # sequence_graph = sequence_graph.transpose(1, 0)
        return len(sequence_feature), sequence_feature, sequence_graph, protein
    else:
        sequence_graph = np.argwhere(sequence_graph > 0)
        np.ravel_multi_index([0, 0], sequence_graph)
        sequence_graph = sequence_graph.transpose(1, 0)

        return len(sequence_feature), sequence_feature, sequence_graph, protein


if __name__ == '__main__':
    sequence_name = 'acpS'
    sequence = 'MAILGLGTDIVEIARIEAVIARSGDRLARRVLSDNEWAIWKTHHQPVRFLAKRFAVKEAAAKAFGTGIRNGLAFNQFEVFNDELGKPRLRLWGEALKLAEKLGVANMHVTLADERHYACATVIIES'
    c_size, features, edge_index, atoms = protein_to_graph(sequence_name, sequence, 0.5)
    g = load_gobal(sequence_name)
    print(atoms)
    print(c_size)
    print(features.shape)
    print(edge_index.shape)

