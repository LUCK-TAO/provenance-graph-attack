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
from src.GCL.Utils.utils import MemoryMonitor

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


def augment_graph(graph, node_drop_rate, edge_drop_rate, add_node_rate, add_edge_rate, feature_noise_level, seed=None):
    if seed is not None:
        random.seed(seed)

    g = graph[0].clone()
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    node_features = g.ndata['type'].clone()
    edge_features = g.edata['type'].clone()

    # 随机删除边
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


    num_new_nodes = int(num_nodes * add_node_rate)
    if num_new_nodes > 0:
        new_node_indices = torch.randint(0, node_features.size(0), (num_new_nodes,))
        new_node_features = node_features[new_node_indices]
        g.add_nodes(num_new_nodes)
        current_node_features = g.ndata['type']
        g.ndata['type'] = torch.cat([current_node_features[:-num_new_nodes], new_node_features], dim=0)


    num_new_edges = int(num_edges * add_edge_rate)
    if num_new_edges > 0:
        new_src = torch.randint(0, g.num_nodes(), (num_new_edges,))
        new_dst = torch.randint(0, g.num_nodes(), (num_new_edges,))
        new_edge_indices = torch.randint(0, edge_features.size(0), (num_new_edges,))
        new_edge_features = edge_features[new_edge_indices]
        g.add_edges(new_src, new_dst)
        current_edge_features = g.edata['type']
        g.edata['type'] = torch.cat([current_edge_features[:-num_new_edges], new_edge_features], dim=0)

    if feature_noise_level > 0:
        num_node_features = node_features.size(1) if len(node_features.size()) > 1 else 1
        num_edge_features = edge_features.size(1) if len(edge_features.size()) > 1 else 1


        num_node_to_perturb = int(num_nodes * feature_noise_level)
        num_edge_to_perturb = int(num_edges * feature_noise_level)

        if num_node_to_perturb > 0:
            node_indices = torch.arange(num_nodes)
            perturb_nodes = torch.randint(0, num_nodes, (num_node_to_perturb,))
            swap_with_nodes = torch.randint(0, num_nodes, (num_node_to_perturb,))

            while any(perturb_nodes == swap_with_nodes):
                swap_with_nodes = torch.randint(0, num_nodes, (num_node_to_perturb,))
            for idx, swap_with_idx in zip(perturb_nodes, swap_with_nodes):
                if swap_with_idx < node_features.size(0) and idx < g.ndata['type'].size(0):
                    g.ndata['type'][idx] = node_features[swap_with_idx]

        if num_edge_to_perturb > 0:
            edge_indices = torch.arange(num_edges)
            perturb_edges = torch.randint(0, num_edges, (num_edge_to_perturb,))
            swap_with_edges = torch.randint(0, num_edges, (num_edge_to_perturb,))

            while any(perturb_edges == swap_with_edges):
                swap_with_edges = torch.randint(0, num_edges, (num_edge_to_perturb,))
            for idx, swap_with_idx in zip(perturb_edges, swap_with_edges):
                if swap_with_idx < edge_features.size(0) and idx < g.edata['type'].size(0):
                    g.edata['type'][idx] = edge_features[swap_with_idx]

    return g


def augment_graph_pair(graph, node_drop_rate=0.7, edge_drop_rate=0.7, add_node_rate=0.7, add_edge_rate=0.7,
                       feature_noise_level=0.7):
    """
    Augments a pair of graphs by applying structural and feature perturbations.

    Parameters:
    - graph: Tuple containing a single graph (and optionally its label, though it is not used here).
    - node_drop_rate: Proportion of nodes to randomly drop.
    - edge_drop_rate: Proportion of edges to randomly drop.
    - add_node_rate: Proportion of new nodes to randomly add.
    - add_edge_rate: Proportion of new edges to randomly add.
    - feature_noise_level: Proportion of features to randomly perturb by swapping with other nodes' or edges' features.

    Returns:
    - Tuple of augmented graphs (g1, g2).
    """
    seed = random.randint(0, 2 ** 32 - 1)  
    g1 = augment_graph(graph, node_drop_rate, edge_drop_rate, add_node_rate, add_edge_rate, feature_noise_level, seed)
    g2 = augment_graph(graph, node_drop_rate, edge_drop_rate, add_node_rate, add_edge_rate, feature_noise_level, seed)
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
    aug_time = 0




    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch_indices in enumerate(train_loader):
            batch_g1 = []
            batch_g2 = []
            start_time = time.time()
            for idx in batch_indices:
                g1, g2 = augment_graph_pair(graphs[idx])
                batch_g1.append(g1.to(device))
                batch_g2.append(g2.to(device))
            end_time = time.time()
            training_time = end_time - start_time
            aug_time = aug_time  + training_time

            batch_g1 = dgl.batch(batch_g1)
            batch_g2 = dgl.batch(batch_g2)

            x1 = batch_g1.ndata['type'].to(torch.long) 
            e1 = batch_g1.edata['type'].to(torch.long)  
            x2 = batch_g2.ndata['type'].to(torch.long)
            e2 = batch_g2.edata['type'].to(torch.long)


            x1 = F.one_hot(x1, num_classes=n_dim).to(torch.float32)
            e1 = F.one_hot(e1, num_classes=e_dim).to(torch.float32)
            x2 = F.one_hot(x2, num_classes=n_dim).to(torch.float32)
            e2 = F.one_hot(e2, num_classes=e_dim).to(torch.float32)


            p1, p2 = model(batch_g1, x1, e1, batch_g2, x2, e2)
            y = torch.ones(p1.size(0)).to(device)
            loss = criterion(p1, p2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g1, batch_g2
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    print(f"Augment_graph took {aug_time:.2f} seconds")
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
    max_epoch = 3
    batch_size = 1


    set_random_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    encoder = GCNEncoder(node_feature_dim, hidden_dim, edge_feature_dim).to(device)
    model = GraphCL(encoder, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    train_loader = extract_dataloaders(train_index,batch_size)


    model = batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, node_feature_dim, edge_feature_dim)

    torch.save(model.state_dict(), f"../data/res_models/graphcl_model_{name}.pth")
    print(f"Model saved to ../data/res_models/graphcl_model_{name}.pth")


if __name__ == "__main__":
    monitor = MemoryMonitor(interval=1)
    monitor.start()

    name = 'wget'
    start_time = time.time()
    main(name)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training with augmentation ratios took {training_time:.2f} seconds")

    monitor.stop()
    avg_memory_usage = monitor.get_average_memory_usage() / (1024 * 1024)  
    peak_memory_usage = monitor.get_peak_memory_usage() / (1024 * 1024)  
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory_usage:.2f} MB")