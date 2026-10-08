import os
import pickle as pkl
import random
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.base import BaseEstimator, ClusterMixin

from model.model import build_model
from Utils.loaddata import load_batch_level_dataset, transform_graph,load_metadata
from Utils.main_config import build_args
from Utils.poolers import Pooling
import pandas as pd
import seaborn as sns
from Utils.utils import load_graph_dataset, set_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def malicious_feature_evasion(graph, malicious_nodes, num_nodes_to_modify):
    """
    改变恶意实体的特征使其模仿良性实体
    :param graph: DGLGraph 对象
    :param malicious_nodes: 恶意节点的索引
    :param num_nodes_to_modify: 要修改特征的恶意节点的数量
    :return: 增强后的图
    """
    malicious_set = set(malicious_nodes)
    all_nodes = set(range(graph.number_of_nodes()))
    benign_nodes = list(all_nodes - malicious_set)

    selected_malicious_nodes = random.sample(list(malicious_nodes), num_nodes_to_modify)
    benign_features = graph.ndata['attr'][benign_nodes].mean(dim=0)
    graph.ndata['attr'][selected_malicious_nodes] = benign_features
    return graph


def malicious_feature_evasion_graph(graph, rate=0.1):
    """
    按比例随机交换两个节点的特征。
    参数:
        graph: DGL 图对象
        rate: 特征交换的比例，范围为 [0, 1]，表示需要交换的节点对数量与节点总数的比例
    返回:
        修改后的 DGL 图
    """
    num_nodes = graph.number_of_nodes()

    # 计算需要交换的节点对数量
    num_swaps = int(num_nodes * rate)

    # 初始化一个集合来存储已交换的节点对（防止重复交换）
    swapped_pairs = set()

    for _ in range(num_swaps):
        while True:
            # 随机选择两个节点
            node_a = random.randint(0, num_nodes - 1)
            node_b = random.randint(0, num_nodes - 1)

            if node_a != node_b and (node_a, node_b) not in swapped_pairs and (node_b, node_a) not in swapped_pairs:
                # 交换节点特征
                temp = graph.ndata['attr'][node_a].clone()
                graph.ndata['attr'][node_a] = graph.ndata['attr'][node_b]
                graph.ndata['attr'][node_b] = temp

                # 记录已交换的节点对
                swapped_pairs.add((node_a, node_b))
                break

    return graph



def malicious_structure_evasion(graph, malicious_nodes, num_nodes_to_modify, num_edges_per_node):
    """
    在恶意实体和良性实体之间选择性添加新边
    :param graph: DGLGraph 对象
    :param malicious_nodes: 恶意节点的索引
    :param num_nodes_to_modify: 要添加边缘的恶意节点的数量
    :param num_edges_per_node: 每个恶意节点添加的边数
    :return: 增强后的图
    """
    malicious_set = set(malicious_nodes)
    all_nodes = set(range(graph.number_of_nodes()))
    benign_nodes = list(all_nodes - malicious_set)

    selected_malicious_nodes = random.sample(list(malicious_nodes), num_nodes_to_modify)
    src = []
    dst = []
    for m_node in selected_malicious_nodes:
        selected_benign_nodes = random.sample(benign_nodes, min(num_edges_per_node, len(benign_nodes)))
        for b_node in selected_benign_nodes:
            src.append(m_node)
            dst.append(b_node)
            src.append(b_node)
            dst.append(m_node)
    src = torch.tensor(src, dtype=torch.int64).to(graph.device)  # 移动到与图相同的设备
    dst = torch.tensor(dst, dtype=torch.int64).to(graph.device)  # 移动到与图相同的设备
    graph.add_edges(src, dst)
    return graph

def malicious_structure_evasion_graph(graph, rate=0.1):
    """
    在恶意实体和良性实体之间选择性添加新边
    """
    # 获取图中所有节点的数量和当前边的数量
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # 计算需要添加的边数量
    num_edges_to_add = int(num_edges * rate)

    # 创建一个集合存储现有的边（避免重复添加）
    existing_edges = set(zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()))

    # 初始化一个空的边列表
    new_edges = []

    for _ in range(num_edges_to_add):
        while True:
            # 随机选择两个节点
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)

            # 确保不添加自环边或重复边
            if src != dst and (src, dst) not in existing_edges:
                new_edges.append((src, dst))
                existing_edges.add((src, dst))  # 添加到现有边集合，避免重复
                break

    # 获取图的设备信息（确保新边与图在同一设备上）
    device = graph.device

    # 将新边添加到图中
    if len(new_edges) > 0:
        src_nodes, dst_nodes = zip(*new_edges)
        graph.add_edges(torch.tensor(src_nodes, device=device), torch.tensor(dst_nodes, device=device))

    return graph

def combined_evasion(graph, malicious_nodes, num_nodes_to_modify_features, num_nodes_to_modify_edges, num_edges_per_node):
    """
    :param graph: DGLGraph 对象
    :param malicious_nodes: 恶意节点的索引
    :param num_nodes_to_modify_features: 要修改特征的恶意节点的数量
    :param num_nodes_to_modify_edges: 要添加边缘的恶意节点的数量
    :param num_edges_per_node: 每个恶意节点添加的边数
    :return: 增强后的图
    """
    graph = malicious_feature_evasion(graph, malicious_nodes, num_nodes_to_modify_features)
    graph = malicious_structure_evasion(graph, malicious_nodes, num_nodes_to_modify_edges, num_edges_per_node)
    return graph

def combined_evasion_graph(graph,rate=0.1):
    """
    :param graph: DGLGraph 对象
    :param malicious_nodes: 恶意节点的索引
    :param num_nodes_to_modify_features: 要修改特征的恶意节点的数量
    :param num_nodes_to_modify_edges: 要添加边缘的恶意节点的数量
    :param num_edges_per_node: 每个恶意节点添加的边数
    :return: 增强后的图
    """
    graph = malicious_feature_evasion_graph(graph, rate/2)
    graph = malicious_structure_evasion_graph(graph, rate/2)
    return graph


def malicious_feature_evasion_pro(graph, malicious_nodes, proportion_to_modify):
    """
    :param graph: DGLGraph 对象
    :param malicious_nodes: 恶意节点的索引
    :param proportion_to_modify: 要修改特征的恶意节点的比例 (0 到 1 之间的浮点数)
    :return: 增强后的图
    """
    malicious_set = set(malicious_nodes)
    all_nodes = set(range(graph.number_of_nodes()))
    benign_nodes = list(all_nodes - malicious_set)

    # 根据比例计算要修改的恶意节点数量
    num_nodes_to_modify = int(len(malicious_nodes) * proportion_to_modify)

    # 随机选择要修改的恶意节点
    selected_malicious_nodes = random.sample(list(malicious_nodes), num_nodes_to_modify)

    # 计算良性节点特征的均值
    benign_features = graph.ndata['attr'][benign_nodes].mean(dim=0)

    # 将选中的恶意节点特征替换为良性节点特征的均值
    graph.ndata['attr'][selected_malicious_nodes] = benign_features

    return graph

def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    # n_neighbors = min(int(train_count * 0.02), 10)
    n_neighbors = 2
    k = 6
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    prec_list = []
    rec_list = []
    f1_list = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    auc_list = []
    accuracy_list = []

    np.random.shuffle(benign_idx)
    np.random.shuffle(attack_idx)
    x_train = x[benign_idx[:train_count]]
    x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
    y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_train)
    cluster_centers = kmeans.cluster_centers_  # 获取聚类中心

    # 计算训练集的平均簇内距离
    train_distances = np.min(np.linalg.norm(x_train[:, None] - cluster_centers, axis=2), axis=1)
    D_mean = train_distances.mean()  # 计算所有样本到其最近簇中心的平均距离

    # 计算测试样本的异常分数
    test_distances = np.min(np.linalg.norm(x_test[:, None] - cluster_centers, axis=2), axis=1)
    score = test_distances / D_mean  # 归一化异常分数

    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    max_f1_idx = np.argmax(f1)
    best_thres = threshold[max_f1_idx]
    prec_list.append(prec[max_f1_idx])
    rec_list.append(rec[max_f1_idx])
    f1_list.append(f1[max_f1_idx])
    print("Best_thres:" + str(best_thres))

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)
    tn_list.append(tn)
    auc_list.append(auc)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy_list.append(accuracy)

    # 根据标签区分良性与恶意的分数
    benign_scores = score[y_test == 0.0]
    malicious_scores = score[y_test == 1.0]

    # 设置 Seaborn 样式
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(10, 6))
    # 绘制良性样本分布，设置直方图与核密度估计
    sns.histplot(benign_scores, bins=30, kde=True, stat="density",
                 color="blue", alpha=0.5, label="Benign")

    sns.histplot(malicious_scores, bins=30, kde=True, stat="density",
                 color="red", alpha=0.5, label="Malicious")


    plt.ylabel(f"Density ({dataset})", fontsize=20)
    # plt.title("Distribution of Benign and Malicious Scores")
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.savefig(f"../vis/sp_{dataset}.png", dpi=450)
    plt.show()


    # 计算 ROC 曲线数据
    fpr, tpr, thresholds_roc = roc_curve(y_test, score)

    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc.to_csv(f'roc_{dataset}_data.csv', index=False)

    print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
    print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
    print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
    print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
    print('ACCURACY: {}+{}'.format(np.mean(accuracy_list), np.std(accuracy_list)))
    print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
    print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
    print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
    print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))

    return np.mean(auc_list), np.std(auc_list)

def evaluate_entity_level_using_knn_kmeans_centers(dataset, x_train, x_test, y_test,n_clusters):



    params_path = f'../data/eval_result/{dataset}_trained_params.pkl'

    if os.path.exists(params_path):
        # 加载已有的训练参数
        with open(params_path, 'rb') as f:
            params = pkl.load(f)
        cluster_centers = params['cluster_centers']
        global_mean_distance = params['global_mean_distance']
        x_train_mean = params['x_train_mean']
        x_train_std = params['x_train_std']
        print("加载训练参数：", params_path)
    else:
        # 训练阶段：计算归一化参数
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train_norm = (x_train - x_train_mean) / x_train_std

        # 使用 KMeans 对训练数据进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(x_train_norm)
        cluster_centers = kmeans.cluster_centers_

        # 计算训练集中每个样本到其所属簇中心的欧氏距离，并取均值
        train_labels = kmeans.predict(x_train_norm)
        distances_train = np.linalg.norm(x_train_norm - cluster_centers[train_labels], axis=1)
        global_mean_distance = distances_train.mean()

        # 保存训练参数
        params = {
            'cluster_centers': cluster_centers,
            'global_mean_distance': global_mean_distance,
            'x_train_mean': x_train_mean,
            'x_train_std': x_train_std
        }
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, 'wb') as f:
            pkl.dump(params, f)
        print("训练阶段完成，参数已保存到：", params_path)


    # 检测阶段：对测试数据使用训练时的均值和标准差进行归一化
    x_test_norm = (x_test - x_train_mean) / x_train_std

    # 进行 PCA 降维到 2D 以便可视化
    pca = PCA(n_components=2)
    x_test_2d = pca.fit_transform(x_test_norm)
    cluster_centers_2d = pca.transform(cluster_centers)

    # 可视化
    plt.figure(figsize=(7, 6))

    sns.set(style="whitegrid", font_scale=1.2)
    # 绘制 KMeans 质心
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], color='darkblue', marker='o', edgecolors='black', s=200,
                label='KMeans Centers')
    # 绘制恶意节点
    plt.scatter(x_test_2d[y_test == 1, 0], x_test_2d[y_test == 1, 1], color='red',edgecolor='k', label='Malicious Nodes')
    print("number of M:")
    print(len(x_test_2d[y_test == 1, :]))



    plt.legend()
    plt.legend(fontsize=14)
    plt.savefig(f'../vis/{dataset}_GMA.png', dpi=400, bbox_inches='tight')
    plt.show()

    # 计算每个测试样本与所有簇中心的距离，取最小距离并归一化得到异常分数
    test_scores = []
    for i in range(len(x_test_norm)):
        distances = np.linalg.norm(x_test_norm[i] - cluster_centers, axis=1)
        min_distance = distances.min()
        score = min_distance / global_mean_distance
        test_scores.append(score)
    test_scores = np.array(test_scores)

    # 评价指标计算
    auc = roc_auc_score(y_test, test_scores)
    prec, rec, threshold = precision_recall_curve(y_test, test_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1)
    best_thres = threshold[best_idx]


    print("Best_thres:"+str(best_thres))
    # 统计混淆矩阵指标
    tn = fn = tp = fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and test_scores[i] >= best_thres:
            tp += 1
        elif y_test[i] == 1.0 and test_scores[i] < best_thres:
            fn += 1
        elif y_test[i] == 0.0 and test_scores[i] < best_thres:
            tn += 1
        elif y_test[i] == 0.0 and test_scores[i] >= best_thres:
            fp += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 根据标签区分良性与恶意的分数
    benign_scores = test_scores[y_test == 0.0]
    malicious_scores = test_scores[y_test == 1.0]

    # 设置 Seaborn 样式
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(10, 6))
    # 绘制良性样本分布，设置直方图与核密度估计
    sns.histplot(benign_scores, bins=30, kde=True, stat="density",
                 color="blue", alpha=0.5, label="Benign")
    # 绘制恶意样本分布
    sns.histplot(malicious_scores, bins=30, kde=True, stat="density",
                 color="red", alpha=0.5, label="Malicious")

    plt.xlabel("Normalized Score")
    plt.ylabel(f"Density ({dataset})",fontsize=20)
    # plt.title("Distribution of Benign and Malicious Scores")
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.savefig(f"../vis/sp_{dataset}.png", dpi=450)
    plt.show()

    # 计算 ROC 曲线数据
    fpr, tpr, thresholds_roc = roc_curve(y_test, test_scores)

    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc.to_csv(f'roc_{dataset}_data.csv', index=False)


    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('ACCURACY: {}'.format(accuracy))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))


    return auc, 0.0, test_scores

def main(main_args):
    logging.info('Starting main function')
    device = main_args.device if main_args.device >= 0 else "cuda"
    logging.info(f'Device: {device}')
    device = torch.device(device)
    dataset_name = main_args.dataset
    attack_type = main_args.attack_type  # 新增参数
    set_random_seed(0)

    if dataset_name not in ['streamspot', 'wget']:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(
            torch.load("../data/res_models/graphcl_model_{}.pth".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()

        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_graph_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g.to('cuda'), g.ndata['attr'].to('cuda'),
                                           g.edata['attr'].to('cuda')).detach().cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)

            x_test = []

            current_index = 0
            for i in range(n_test):
                g = load_graph_dataset(dataset_name, 'test', i).to(device)
                # 获取该图的恶意节点索引
                graph_malicious_nodes = [node for node in malicious if node-current_index < g.number_of_nodes() and node-current_index >0 ]
                malicious_node_index = [node - current_index for node in graph_malicious_nodes]
                current_index = current_index + g.number_of_nodes()

                if attack_type == 'GFPA':
                    g = malicious_feature_evasion(g, malicious_node_index, round(len(graph_malicious_nodes) * args.ratio))
                elif attack_type == 'GSPA':
                    g = malicious_structure_evasion(g, malicious_node_index, round(len(graph_malicious_nodes) * args.ratio), 1)
                elif attack_type == 'CGPA':
                    g = combined_evasion(g, malicious_node_index, round(len(graph_malicious_nodes) * (args.ratio/2)),
                                         round(len(graph_malicious_nodes) * (args.ratio/2)), 1)

                x_test.append(model.embed(g.to('cuda'), g.ndata['attr'].to('cuda'),
                                          g.edata['attr'].to('cuda')).detach().cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            y_test = np.zeros(x_test.shape[0])
            y_test[malicious] = 1.0

            # evaluate_entity_level_using_knn_kmeans_centers(dataset_name,x_train, x_test, y_test, args.clusters)
            evaluate_entity_level_using_knn_kmeans_centers(dataset_name,x_train, x_test, y_test,100)

    else:
        data = load_batch_level_dataset(dataset_name)

        main_args.n_dim = data['n_feat']
        main_args.e_dim = data['e_feat']
        model = build_model(main_args)
        # model.load_state_dict(torch.load("../data/res_models/dgi_model_{}.pth".format(dataset_name), map_location=device))
        model.load_state_dict(torch.load("../data/res_models/graphcl_model_{}.pth".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        model.eval()
        x_list = []
        y_list = []

        full = data['full_index']
        graphs = data['dataset']
        dataset = dataset_name
        with torch.no_grad():
            for i in full:
                g = transform_graph(graphs[i][0], main_args.n_dim, main_args.e_dim).to(device)
                label = graphs[i][1]

                if attack_type == 'GFPA':
                    g = malicious_feature_evasion_graph(g, 0.5)
                elif attack_type == 'GSPA':
                    g = malicious_structure_evasion_graph(g, 0.5)
                elif attack_type == 'CGPA':
                    g = combined_evasion_graph(g,0.5)
                out = model.embed(g, g.ndata['attr'], g.edata['attr'])
                if dataset != 'wget':
                    out = pooler(g, out).cpu().numpy()
                else:
                    out = pooler(g, out, [2]).cpu().numpy()
                y_list.append(label)
                x_list.append(out)
        x = np.concatenate(x_list, axis=0)
        y = np.array(y_list)

        # 使用 t-SNE 进行降维（从高维到2维）
        tsne = TSNE(n_components=2, random_state=42)
        x_tsne = tsne.fit_transform(x)

        # 可视化结果
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='coolwarm', marker='o', edgecolor='k', alpha=0.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malicious'], loc="upper right", fontsize=14)
        plt.savefig('../vis/scatter_plot.png', dpi=400, bbox_inches='tight')
        plt.show()

        if dataset == 'streamspot':
            train_count = 400
        else:
            train_count = 100
        # n_neighbors = min(int(train_count * 0.02), 10)
        benign_idx = np.where(y == 0)[0]
        attack_idx = np.where(y == 1)[0]
        set_random_seed(2022)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)

        x_train_std[x_train_std == 0] = 1


        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        if dataset == 'streamspot':
            evaluate_batch_level_using_knn(-1, dataset, x, y)
        else:
            evaluate_batch_level_using_knn(-1, dataset, x, y)
        return

if __name__ == '__main__':


    start_time = time.time()

    args = build_args()
    args.num_hidden = 64
    args.dataset = 'cadets'
    args.ratio = 0
    args.attack_type = 'GSPA'
    args.attack_rate = 0.3
    main(args)

    # args = build_args()
    # args.num_hidden = 64
    # args.dataset = 'trace'
    # args.ratio = 0
    # args.attack_type = 'null'
    # main(args)
    #
    #
    # args = build_args()
    # args.num_hidden = 64
    # args.dataset = 'theia'
    # args.ratio = 0
    # args.attack_type = 'null'
    # main(args)
    #
    # args = build_args()
    # args.dataset = 'streamspot'
    # args.num_hidden = 256
    # args.attack_type = 'null'
    # main(args)
    #
    #
    #
    # args = build_args()
    # args.dataset = 'wget'
    # args.num_hidden = 256
    # args.attack_type = 'null'
    # main(args)


    end_time = time.time()
    training_time = end_time - start_time
    print(f"Detection took {training_time:.2f} seconds")




