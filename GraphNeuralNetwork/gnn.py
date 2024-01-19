import os
import sys
import joblib
import time
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import torch
import networkx as nx
import scipy as sp
from scipy.sparse import coo_array
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
from sklearn.datasets import fetch_20newsgroups
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from text2graphapi.src.Cooccurrence  import Cooccurrence
from text2graphapi.src.Heterogeneous  import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph  import ISG


#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#************************************* CONSTANTS

LANGUAGE = 'en' #es, en, fr
DATA_DIR_PATH = 'C:/Users/anvaldez/Documents/Docto/Projects/SemEval/GraphNeuralNetwork/data/'
CUT_PERCENTAGE_DATASET = 1

#************************************* UTILS

def print_graph_metrics(graph_output):
    # get and save metrics
    print("*** TEST: Results - Metrics ")
    print('\t * Num_Graph_Docs_Output: %i', len(graph_output))
    # show corpus_graph_docs
    for g in graph_output[:5]:
        print('graph: ', str(g))
        #print("nodes: ", g['graph'].nodes(data=True))
        #print("edges: ", g['graph'].edges(data=True))

def save_data(data, file_name, path=DATA_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Saving data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path=DATA_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Loading data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    return joblib.load(path_file)

def print_dataloader_info(dataset):
    print(dataset)
    print("Dataset type: ", type(dataset))
    print("Dataset features: ", dataset.num_features)
    print("Dataset target: ", dataset.num_classes)
    print("Dataset length: ", dataset.len)
    print("Dataset sample: ", dataset[0])
    print("Sample  nodes: ", dataset[0].num_nodes)
    print("Sample  edges: ", dataset[0].num_edges)
    #print(dataset[0].edge_index.t())
    #print(dataset[0].y)
    #print(dataset[0].x)

def build_vocab(graphs_data):
    vocab = set()
    for g in graphs_data:
        for node in list(g['graph'].nodes):
            vocab.add(node)
    print('Vocab Len: ', len(vocab))
    return vocab

#************************************* TEXT 2 GRAPH CONFIG

def cooccur_graph_instance(lang='en'):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'DiGraph',
            window_size = 2,
            apply_prep = True,
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": True,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_contractions": True,
                "handle_stop_words": True,
                "to_lowercase": True
            },
            parallel_exec = False,
            language = lang, #es, en, fr
            output_format = 'networkx'
        )
    return co_occur

def hetero_graph_instance(lang='en'):
    # create co_occur object
    hetero_graph = Heterogeneous(
        graph_type = 'DiGraph',
        window_size = 10,
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        load_preprocessing = False,
        language = lang, #sp, en, fr
        output_format = 'networkx',
    )
    return hetero_graph

def isg_graph_instance(lang='en'):
    # create isg object
    isg = ISG(
        graph_type = 'DiGraph',
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        language = lang, #spanish (sp), english (en), french (fr)
        output_format = 'networkx'
    )
    return isg


#************************************* DATASETS
def read_20_newsgroups_dataset(subset='train'):
    newsgroups_dataset = fetch_20newsgroups(subset=subset) #subset='train', fetch from sci-kit learn
    id = 1
    corpus_text_docs = []
    for index in range(len(newsgroups_dataset.data)):
        doc = {"id": id, "doc": newsgroups_dataset.data[index], "context": {"target": newsgroups_dataset.target[index]}}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs

def read_semeval_dataset():
  ...


#************************************* TEXT 2 GRAPH TRANSFORMATION

def transform_text_to_graph(corpus_text_docs):
    print("Init transform text to graph: ")
    t2graph = cooccur_graph_instance()
    #t2graph = hetero_graph_instance()
    #t2graph = isg_graph_instance()

    # Apply t2g transformation
    cut_dataset = len(corpus_text_docs) * (int(CUT_PERCENTAGE_DATASET) / 100)
    start_time = time.time() # time init
    graph_output = t2graph.transform(corpus_text_docs[:int(cut_dataset)])
    for corpus_text_doc in corpus_text_docs:
        for g in graph_output:
            if g['doc_id'] == corpus_text_doc['id']:
                g['context'] = corpus_text_doc['context']
                break
    end_time = (time.time() - start_time)
    print("\t * TOTAL TIME:  %s seconds" % end_time)
    return graph_output


#************************************* DATASET SETUP

class BuildDataset(Dataset):
    def __init__(self, root, filename, vocab, num_instances=-1, test=False, transform=None, pre_transform=None, pre_filter=None):
        self.vocab = vocab
        self.node_to_index = {}
        self.filename = filename
        self.test = test
        self.num_instances = num_instances
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
            node_feats = self.__get_node_features(g['graph'])
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


    def __get_node_features(self, g):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        # get OHE for each node graph
        graph_node_feat = []
        for node in list(g.nodes):
            vector = np.zeros(len(self.vocab))
            vector[self.node_to_index[node]] = 1
            graph_node_feat.append(vector)
        
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
    
#************************************* GNN SETUP

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def train(model, criterion, optimizer, train_loader):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model.forward(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model,loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.




#************************************* MAIN

def main(graph_trans=False, loadata=False):

    if graph_trans:
        corpus_text_docs_train = read_20_newsgroups_dataset(subset='train')
        graph_output_train = transform_text_to_graph(corpus_text_docs_train)
        save_data(graph_output_train, path=DATA_DIR_PATH + 'raw', file_name='graphs_train')
        
        corpus_text_docs_test = read_20_newsgroups_dataset(subset='test')
        graph_output_test = transform_text_to_graph(corpus_text_docs_test)
        save_data(graph_output_test, path=DATA_DIR_PATH + 'raw', file_name='graphs_test')
        
        vocab = build_vocab(graph_output_train + graph_output_test)
        save_data(vocab, path=DATA_DIR_PATH, file_name='vocab')
    if loadata:
        ...
        #graph_output = load_data(path=DATA_DIR_PATH + 'raw', file_name='graphs')
    
    #print_graph_metrics(graph_output_train)
    #print_graph_metrics(graph_output_test)
    vocab = load_data(path=DATA_DIR_PATH, file_name='vocab')
    train_dataset = BuildDataset(root=DATA_DIR_PATH, vocab=vocab, filename='graphs_train.pkl')
    test_dataset = BuildDataset(root=DATA_DIR_PATH, vocab=vocab, filename='graphs_test.pkl', test=True)
    
    print_dataloader_info(train_dataset)
    print_dataloader_info(test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes
    hidden_channels = 64

    model = GCN(num_features = num_features, hidden_channels=hidden_channels, num_classes = num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    print(model)

    for epoch in range(1, 50):
        train(model, criterion, optimizer, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


main(
    graph_trans=True
)
























