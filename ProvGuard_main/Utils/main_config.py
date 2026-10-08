import argparse


def build_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning')


    parser.add_argument('--device', type=int, default=0, help='GPU device id (default: 0)')


    parser.add_argument('--dataset', type=str, default='trace', choices=['trace', 'streamspot', 'wget'],
                        help='Dataset name (default: trace)')


    parser.add_argument('--num_hidden', type=int, default=256, help='Number of hidden units (default: 64)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers (default: 3)')


    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method (default: mean)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--path', type=str, default='null', help='eval path')

    args = parser.parse_args()

    return args

