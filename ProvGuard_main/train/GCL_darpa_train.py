import os
import pickle as pkl
import time

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
import random

from model.model import GraphCL
from Utils.utils import MemoryMonitor
from Utils.loaddata import load_metadata
from model.model import GCNEncoder


def malicious_feature_evasion(graph, num_nodes_to_modify):

    all_nodes = list(range(graph.num_nodes()))
    nodes_to_modify = random.sample(all_nodes, num_nodes_to_modify)
    other_nodes = list(set(all_nodes) - set(nodes_to_modify))
    if not other_nodes:
        return graph


    new_features = graph.ndata['attr'][random.choice(other_nodes)].clone()
    graph.ndata['attr'][nodes_to_modify] = new_features
    return graph



def malicious_structure_evasion(graph, num_nodes_to_modify, num_edges_per_node):


    all_nodes = set(range(graph.num_nodes()))
    nodes_to_modify = random.sample(all_nodes, num_nodes_to_modify)
    benign_nodes = list(all_nodes - set(nodes_to_modify))

    selected_malicious_nodes = random.sample(list(nodes_to_modify), num_nodes_to_modify)
    src = []
    dst = []
    for m_node in selected_malicious_nodes:
        selected_benign_nodes = random.sample(benign_nodes, min(num_edges_per_node, len(benign_nodes)))
        for b_node in selected_benign_nodes:
            src.append(m_node)
            dst.append(b_node)
            src.append(b_node)
            dst.append(m_node)

    graph.add_edges(src, dst)
    return graph



def combined_evasion(graph, num_nodes_to_modify_features, num_nodes_to_modify_edges, num_edges_per_node):

    graph = malicious_feature_evasion(graph,  num_nodes_to_modify_features)
    graph = malicious_structure_evasion(graph, num_nodes_to_modify_edges, num_edges_per_node)
    return graph





class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        euclidean_distance = F.pairwise_distance(z1, z2)
        loss = (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) + y * torch.pow(euclidean_distance, 2)
        return loss.mean()



def load_graphs_from_pkl(path):
    graphs = []
    for i in range(10):
        file_path = os.path.join(path, f'train{i}.pkl')
        if not os.path.exists(file_path):
            # print(f"File {file_path} does not exist. Please check the file path.")
            continue
        with open(file_path, 'rb') as f:
            g = pkl.load(f)
            graphs.append(g)
        print(f"Loaded graph {i+2} from {file_path}")
    return graphs
def augment_graph_darpa(
        graph,
        node_type_map,
        edge_type_map,
        node_drop_rate=0.2,
        edge_drop_rate=0.2,
        add_edge_rate=0.2,
):

    g = graph.clone()
    device = g.device


    if g.num_nodes() > 0 and node_drop_rate > 0:
        drop_cnt = int(g.num_nodes() * node_drop_rate)
        if drop_cnt > 0:

            nodes_to_drop = torch.randperm(g.num_nodes())[:drop_cnt].to(device)

            g = dgl.remove_nodes(g, nodes_to_drop)


    if g.num_edges() > 0 and edge_drop_rate > 0:
        drop_cnt = int(g.num_edges() * edge_drop_rate)
        if drop_cnt > 0:

            edges_to_drop = torch.randperm(g.num_edges())[:drop_cnt].to(device)

            g = dgl.remove_edges(g, edges_to_drop)


    if g.num_nodes() == 0:
        return g


    def find_id(mapping, keyword):
        for k, v in mapping.items():
            if keyword.upper() in k.upper():
                return v
        return None

    proc_id = find_id(node_type_map, 'PROCESS')
    file_id = find_id(node_type_map, 'FILE')
    net_id = find_id(node_type_map, 'NETFLOW') or find_id(node_type_map, 'SOCKET')

    fork_id = find_id(edge_type_map, 'FORK') or find_id(edge_type_map, 'CLONE')
    write_id = find_id(edge_type_map, 'WRITE')
    read_id = find_id(edge_type_map, 'READ')
    send_id = find_id(edge_type_map, 'SEND')
    recv_id = find_id(edge_type_map, 'RECV')

    node_types = g.ndata.get('type', None)
    if node_types is None:
        return g

    proc_indices = torch.nonzero(node_types == proc_id, as_tuple=False).flatten().tolist()
    file_indices = torch.nonzero(node_types == file_id, as_tuple=False).flatten().tolist()
    net_indices = torch.nonzero(node_types == net_id, as_tuple=False).flatten().tolist()


    possible_ops = []
    if proc_indices and fork_id is not None:
        possible_ops.append((proc_indices, proc_indices, fork_id))
    if proc_indices and file_indices:
        if write_id is not None: possible_ops.append((proc_indices, file_indices, write_id))
        if read_id is not None: possible_ops.append((file_indices, proc_indices, read_id))
    if proc_indices and net_indices:
        if send_id is not None: possible_ops.append((proc_indices, net_indices, send_id))
        if recv_id is not None: possible_ops.append((net_indices, proc_indices, recv_id))

    if not possible_ops:
        return g

    new_src, new_dst, new_attr_ids = [], [], []
    edge_attr_dim = g.edata['attr'].shape[1]
    budget = int((g.num_edges() + 1) * add_edge_rate)


    for _ in range(budget):
        s_list, d_list, eid = random.choice(possible_ops)
        s = random.choice(s_list)
        d = random.choice(d_list)
        new_src.append(s)
        new_dst.append(d)
        new_attr_ids.append(eid)

    if new_src:
        new_attr_onehot = F.one_hot(torch.tensor(new_attr_ids), num_classes=edge_attr_dim).float().to(device)
        new_src_t = torch.tensor(new_src).to(device)
        new_dst_t = torch.tensor(new_dst).to(device)


        old_num_edges = g.num_edges()

        g.add_edges(new_src_t, new_dst_t)

  
        g.edata['attr'] = torch.cat([g.edata['attr'][:old_num_edges], new_attr_onehot], dim=0)

    return g

def train_graphcl(graphs, in_dim, edge_dim, hidden_dim, name, lr=0.001, epochs=5, device='cuda',attack_type='null'):
    print("Initializing GCNEncoder and GraphCL res_models...")
    encoder = GCNEncoder(in_dim, hidden_dim, edge_dim).to(device)
    model = GraphCL(encoder, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    contrastive_criterion = ContrastiveLoss(margin=1.0)

    metadata = load_metadata(name)
    malicious, _ = metadata['malicious']
    current_index = 0


    for i, g in enumerate(graphs):

        if attack_type == 'GFPA':
            g = malicious_feature_evasion(g, len(malicious)*0.2)
        elif attack_type == 'GSPA':
            g = malicious_structure_evasion(g, len(malicious)*0.2, 1)
        elif attack_type == 'CGPA':
            g = combined_evasion(g, len(malicious)*0.2, len(malicious)*0.2, 1)
        graphs[i] = g

    all_aug_time = 0


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print(f"Starting epoch {epoch+1}/{epochs}...")
        for i, g in enumerate(graphs):


            start_time = time.time()
            graph_malicious_nodes = [node for node in malicious if
                                     node - current_index < g.number_of_nodes() and node - current_index > 0]
            malicious_node_index = [node - current_index for node in graph_malicious_nodes]
            current_index = current_index + g.number_of_nodes()

            metadata = load_metadata(name)
            node_type_map = metadata['node_type_map']
            edge_type_map = metadata['edge_type_map']

            g1 = augment_graph_darpa(g, node_type_map, edge_type_map)
            g2 = augment_graph_darpa(g, node_type_map, edge_type_map)
            end_time = time.time()
            aug_time = end_time - start_time
            all_aug_time = all_aug_time+aug_time

            g1 = g1.to(device)
            g2 = g2.to(device)

            x1 = g1.ndata['attr'].to(device)
            e1 = g1.edata['attr'].to(device)
            x2 = g2.ndata['attr'].to(device)
            e2 = g2.edata['attr'].to(device)

            z1, z2 = model(g1, x1, e1, g2, x2, e2)

            # Correctly construct labels to match the size of z1 and z2
            labels = torch.ones(z1.size(0)).to(device)  # 相似样本对标签为1

            loss = contrastive_criterion(z1, z2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Graph {i}: Loss = {loss.item()}")

            del g1, g2, x1, e1, x2, e2, z1, z2, labels, loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{epochs} completed: Average Loss = {epoch_loss/len(graphs)}")
    print(f"Training took {all_aug_time:.2f} seconds")
    return model

# 使用示例
def main(name):
    path = '../data/{}'.format(name)

    metadata = load_metadata(name)

    malicious, _ = metadata['malicious']

    node_feature_dim = metadata['node_feature_dim']
    edge_feature_dim = metadata['edge_feature_dim']

    graphs = load_graphs_from_pkl(path)

    hidden_dim = 64

    print("Starting training...")
    model = train_graphcl(graphs, node_feature_dim, edge_feature_dim, hidden_dim, device='cuda', name=name,attack_type='null')




    model_save_path = "../data/res_models/graphcl_model_{}.pth".format(name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")



if __name__ == "__main__":

    monitor = MemoryMonitor(interval=1)
    monitor.start()
    name = 'cadets'
    start_time = time.time()
    main(name)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training took {training_time:.2f} seconds")

    monitor.stop()
    avg_memory_usage = monitor.get_average_memory_usage() / (1024 * 1024)  # 转换为 MB
    peak_memory_usage = monitor.get_peak_memory_usage() / (1024 * 1024)  # 转换为 MB
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory_usage:.2f} MB")
