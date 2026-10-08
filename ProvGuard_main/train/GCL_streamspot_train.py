import os
import json
import pickle as pkl
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.dataloading import GraphDataLoader
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from dgl import DGLGraph
import dgl

# Utils
from src.GCL.Utils.loaddata import load_rawdata


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader




def load_batch_level_dataset(dataset_name):
    dataset = load_rawdata(dataset_name)
    graph, _ = dataset[0]
    node_feature_dim = 0
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())
    edge_feature_dim = 0
    for g, _ in dataset:
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())
    node_feature_dim += 1
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}

def augment_graph(graph, node_drop_rate, edge_drop_rate, feature_noise_level, seed=None):
    if seed is not None:
        random.seed(seed)

    g = graph[0].clone()
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    # print(f"Original num_nodes: {num_nodes}, num_edges: {num_edges}")

    node_features = g.ndata['type'].clone().to(torch.float32)
    edge_features = g.edata['type'].clone().to(torch.float32)

 
    num_edges_to_drop = int(num_edges * edge_drop_rate)
    if num_edges_to_drop > 0 and num_edges > num_edges_to_drop:
        edges_to_drop = random.sample(range(num_edges), num_edges_to_drop)
        edge_mask = torch.ones(num_edges, dtype=torch.bool)
        edge_mask[edges_to_drop] = False
        src, dst = g.edges()
        src = src[edge_mask]
        dst = dst[edge_mask]
        g = dgl.graph((src, dst), num_nodes=g.num_nodes())
        g.edata['type'] = edge_features[edge_mask]
   

    num_nodes_to_drop = int(num_nodes * node_drop_rate)
    if num_nodes_to_drop > 0 and num_nodes > num_nodes_to_drop:
        nodes_to_drop = random.sample(range(num_nodes), num_nodes_to_drop)
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_mask[nodes_to_drop] = False
        g = dgl.node_subgraph(g, node_mask)
        g.ndata['type'] = node_features[node_mask]



    return g


def augment_graph_pair(graph, node_drop_rate=0.3, edge_drop_rate=0.3, feature_noise_level=0.3):
    seed = random.randint(0, 2 ** 32 - 1)
    g1 = augment_graph(graph, node_drop_rate, edge_drop_rate, feature_noise_level, seed)
    g2 = augment_graph(graph, node_drop_rate, edge_drop_rate, feature_noise_level, seed)
    return g1, g2


# Model definition
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g, x, e):
        e = self.edge_mlp(e)
        g.edata['e'] = e
        x = F.relu(self.conv1(g, x))
        x = x.mean(dim=1)
        x = self.conv2(g, x)
        x = x.mean(dim=1)
        return x

class GraphCL(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(GraphCL, self).__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g1, x1, e1, g2, x2, e2):
        z1 = self.encoder(g1, x1, e1)
        z2 = self.encoder(g2, x2, e2)
        p1 = self.projection(z1)
        p2 = self.projection(z2)
        return p1, p2

    def embed(self, g, x, e):
        return self.encoder(g, x, e)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        euclidean_distance = F.pairwise_distance(z1, z2)
        # print(f"Euclidean distance: {euclidean_distance}")
        loss = (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) + y * torch.pow(euclidean_distance, 2)
        return loss.mean()

def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim, e_dim):
    epoch_iter = tqdm(range(max_epoch))
    criterion = ContrastiveLoss()
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch_indices in enumerate(train_loader):
            batch_g1 = []
            batch_g2 = []
            for idx in batch_indices:
                g1, g2 = augment_graph_pair(graphs[idx])
                batch_g1.append(g1.to(device))
                batch_g2.append(g2.to(device))

            batch_g1 = dgl.batch(batch_g1)
            batch_g2 = dgl.batch(batch_g2)

            x1 = batch_g1.ndata['type'].to(torch.long)  
            e1 = batch_g1.edata['type'].to(torch.long)  #
            x2 = batch_g2.ndata['type'].to(torch.long)
            e2 = batch_g2.edata['type'].to(torch.long)

            # 对特征进行 one-hot 编码
            x1 = F.one_hot(x1, num_classes=n_dim).to(torch.float32)
            e1 = F.one_hot(e1, num_classes=e_dim).to(torch.float32)
            x2 = F.one_hot(x2, num_classes=n_dim).to(torch.float32)
            e2 = F.one_hot(e2, num_classes=e_dim).to(torch.float32)



            p1, p2 = model(batch_g1, x1, e1, batch_g2, x2, e2)
            y = torch.ones(p1.size(0)).to(device)
            loss = criterion(p1, p2, y)

            # Check loss for debugging
            # print(f"Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g1, batch_g2
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model



# Main function
def main(name):
    path = '../data/{}'.format(name)


    dataset = load_batch_level_dataset(name)
    graphs = dataset['dataset']
    train_index = dataset['train_index']
    node_feature_dim = dataset['n_feat']
    edge_feature_dim = dataset['e_feat']


    hidden_dim = 256

    lr = 0.001
    max_epoch = 5
    batch_size = 12


    set_random_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = GCNEncoder(node_feature_dim, hidden_dim, edge_feature_dim).to(device)
    model = GraphCL(encoder, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    train_loader = extract_dataloaders(train_index,batch_size)

    model = batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, node_feature_dim, edge_feature_dim)

    torch.save(model.state_dict(), f"../data/res_models/graphcl_model_{name}.pth")
    print(f"Model saved to ../data/models/graphcl_model_{name}.pth")


if __name__ == "__main__":
    name = 'streamspot'
    start_time = time.time()
    main(name)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training with augmentation ratios took {training_time:.2f} seconds")