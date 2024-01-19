import os
import torch
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp
from gensim.models import Word2Vec
import gensim

from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader

class BuildDataset(Dataset):
    def __init__(self, root, filename, vocab, model_w2v, num_instances=-1, test=False, transform=None, pre_transform=None, pre_filter=None):
        self.vocab = vocab
        self.node_to_index = {}
        self.filename = filename
        self.test = test
        self.num_instances = num_instances
        self.model_w2v = model_w2v
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.filename 

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        data_len = len(joblib.load(self.raw_paths[0]))
        if self.test:
            return [f'data_test_{i}.pt' for i in range(1,data_len+1)]
        else:
            return [f'data_{i}.pt' for i in range(1,data_len+1)]
        '''
        res = []
        for i in range(1, self.num_instances + 1):
            res.append('data_'+ str(i) + 'pt')
        return res
        '''

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        self.graphs_data = joblib.load(self.raw_paths[0])
        self.graphs_data = self.graphs_data[:self.num_instances]
        #print(type(self.graphs_data), len(self.graphs_data))
        #print(self.graphs_data)

        # get vocab from graph nodes
        '''
        for g in self.graphs_data:
            #print(g)
            for node in list(g['graph'].nodes):
                self.vocab.add(node)
        print('Vocab Len: ', len(self.vocab))
        '''

        # map node to int index
        for i, node in enumerate(self.vocab):
            self.node_to_index[node] = i

            
        # process each graph
        for index, g in enumerate(tqdm(self.graphs_data)):
            #print(index, g)
            # Get node features
            node_feats = self.__get_node_features(g['graph'], type='w2v')
            # Get edge features
            edge_feats = self.__get_edge_features(g['graph'])
            # Get adjacency info
            edge_index = self.__get_adjacency_info(g['graph'])
            # Get labels info
            label = self.__get_labels(g["context"]["target"])
            
            #print(node_feats.shape, edge_index.shape, label.shape)

            data = Data(
                x = node_feats,
                edge_index = edge_index,
                y = label
                #edge_attr = edge_feats,
            )
            
            #data.validate(raise_on_error=False)
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index+1}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index+1}.pt'))

        #if self.pre_filter is not None and not self.pre_filter(data):
        #    continue

        #if self.pre_transform is not None:
        #    data = self.pre_transform(data)

        #torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        #idx += 1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):    
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx+1}.pt'))  
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx+1}.pt'))  
        return data


    def __get_node_features(self, g, type='OHE'):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        # get OHE for each node graph
        if type == 'OHE':
            graph_node_feat = []
            for node in list(g.nodes):
                vector = np.zeros(len(self.vocab))
                vector[self.node_to_index[node]] = 1
                graph_node_feat.append(vector)
        
        if type == 'w2v':
            graph_node_feat = []
            for node in list(g.nodes):
                graph_node_feat.append(self.model_w2v.wv[node])
            
        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float)
    
    def __get_edge_features(self, g):
        return None
    
    def __get_adjacency_info(self, g):
        adj = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.sparse.coo_array(adj)
        edge_indices = []
        for index in range(adj_coo.shape[0]):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices)
        return edge_indices.t().to(torch.long).view(2, -1)
        '''
        adj = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.sparse.coo_array(adj)
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_coo.shape
        return torch.sparse.FloatTensor(i, v, (2,torch.Size(shape)[1]))
        '''
    
    def __get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    