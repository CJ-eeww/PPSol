import os
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
import pandas as pd
import numpy as np
from creat_data_DC import protein_to_graph, load_gobal


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='tmp', dataset='', patt='re', transform=None,
                 pre_transform=None, smile_graph=None, p=0.5):
        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.patt = patt + str(p)
        self.processed_paths[0] = self.processed_paths[0] + self.patt
        self.p = p

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root, self.dataset)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + self.patt + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, root, dataset):
        data_list = []
        dataframe = pd.read_csv('./'+root+'/'+dataset + '.csv', sep=',')
        sequence_names = dataframe['gene'].values
        sequence = dataframe['sequence'].values
        sequence_label = dataframe['solubility'].values

        count = 0
        for protein_name, protein, label in zip(sequence_names, sequence, sequence_label):
            if len(protein) < 4:
                continue
            count = count + 1
            if label > 1:
                label = 1.0
                print(protein_name)
        
            print('protein ', count, protein_name, protein, label)
            x_size, features, edge_index, sequence = protein_to_graph(protein_name, protein, p=self.p)
            global_features = load_gobal(protein_name)
            

            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__('x_size', torch.LongTensor([x_size]))
            GCNData.__setitem__('edge_size', torch.LongTensor([len(edge_index)]))
            GCNData.__setitem__('y', torch.Tensor([label]))
            GCNData.__setitem__('global_features', torch.Tensor([np.squeeze(global_features)]))
            
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
