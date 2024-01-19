
import os
import joblib
import glob
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import configs
import utils

def print_graph_metrics(graph_output):
    # get and save metrics
    print("*** TEST: Results - Metrics ")
    print('\t * Num_Graph_Docs_Output: %i', len(graph_output))
    # show corpus_graph_docs
    for g in graph_output[:5]:
        print('graph: ', str(g))
        #print("nodes: ", g['graph'].nodes(data=True))
        #print("edges: ", g['graph'].edges(data=True))

def save_data(data, file_name, path=configs.DATA_DIR_PATH, format_file='.pkl', compress=False):
    configs.logger.info('Saving data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path=configs.DATA_DIR_PATH, format_file='.pkl', compress=False):
    configs.logger.info('Loading data: %s', file_name)
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

def read_dataset(dataset):
    print(dataset)
    train, test  = [], []
    if dataset == '20ng':
        train = utils.read_20_newsgroups_dataset(subset='train')
        test = utils.read_20_newsgroups_dataset(subset='test')
    elif dataset == 'semeval-taskA-mon':
        train_docs = pd.read_json(path_or_buf=configs.DATASETS_DIR_PATH + 'SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
        test_docs = pd.read_json(path_or_buf=configs.DATASETS_DIR_PATH + 'SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)
        train, test = [], []
        for i, doc in enumerate(list(train_docs.to_dict('records'))):
            train.append({"id": doc['id'], "doc": doc['text'], 'context': {'target': doc['label'], "model": doc["model"],"source": doc["source"]}})
        for i, doc in enumerate(list(test_docs.to_dict('records'))):
            test.append({"id": doc['id'], "doc": doc['text'], 'context': {'target': doc['label'], "model": doc["model"],"source": doc["source"]}})

        return train, test
    elif dataset == 'semeval-taskA-bil':
        ...
    else:
        ...
    
    return train, test

def read_20_newsgroups_dataset(subset='train'):
    newsgroups_dataset = fetch_20newsgroups(subset=subset) #subset='train', fetch from sci-kit learn
    id = 1
    corpus_text_docs = []
    for index in range(len(newsgroups_dataset.data)):
        doc = {"id": id, "doc": newsgroups_dataset.data[index], "context": {"target": newsgroups_dataset.target[index]}}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs
