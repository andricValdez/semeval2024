
import torch
import argparse
from torch_geometric.data import DataLoader
import gensim

import utils
import configs
from model import GCN
from model_train import train, test
from graph_processor import BuildDataset
import text_to_graph

def main(dataset):
    # Read dataset: 20ng, semeval-taskA-mon, semeval-taskA-bil
    train_set, test_set = utils.read_dataset(dataset)
    print(len(train_set), len(test_set))    

    # Transform text-to-graph
    graphs_train_set = text_to_graph.transform(train_set)
    graphs_test_set = text_to_graph.transform(test_set)
    utils.save_data(graphs_train_set, path=configs.DATA_DIR_PATH + 'raw', file_name='graphs_train')
    utils.save_data(graphs_test_set, path=configs.DATA_DIR_PATH + 'raw', file_name='graphs_test')

    # Build vocab
    vocab = utils.build_vocab(graphs_train_set + graphs_test_set)
    utils.save_data(vocab, path=configs.DATA_DIR_PATH, file_name='vocab')
    sent_w2v = []
    for g in graphs_train_set + graphs_test_set:
        sent_w2v.append(list(g['graph'].nodes))
    model_w2v = gensim.models.Word2Vec(sent_w2v, min_count=1,vector_size=300, window=5)

    # Process graphs - Build Dataset
    #vocab = utils.load_data(path=configsDATA_DIR_PATH, file_name='vocab')
    train_dataset = BuildDataset(root=configs.DATA_DIR_PATH, model_w2v=model_w2v, vocab=vocab, filename='graphs_train.pkl')
    test_dataset = BuildDataset(root=configs.DATA_DIR_PATH, model_w2v=model_w2v, vocab=vocab, filename='graphs_test.pkl', test=True)    
    utils.print_dataloader_info(train_dataset)

    # Set Dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Set and Train GNN
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--dataset", help="dataset to use", default='20ng', type=str)
    args, unknown_args = parser.parse_known_args()
    args = vars(args)
    main(dataset=args['dataset'])
    

# Dataset Options: 20ng, semeval-taskA-mon, semeval-taskA-bil