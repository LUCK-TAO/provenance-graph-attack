import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GINConv
from dgl.nn.pytorch import SAGEConv

# GCNEncoder definition
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

# GCNEncoder_hyper definition
class GCNEncoder_hyper(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, num_layers=2):
        super(GCNEncoder_hyper, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(GATConv(input_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True))
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g, x, e):
        e = self.edge_mlp(e)
        g.edata['e'] = e
        for layer in self.layers:
            x = F.relu(layer(g, x))
            x = x.mean(dim=1)
        return x

# GraphCL definition
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

# DGIEncoder definition
class DGIEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(DGIEncoder, self).__init__()
        self.conv = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

    def forward(self, g, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(features.size(0))
            features = features[perm]
        return self.conv(g, features)

class DGIModule(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(DGIModule, self).__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g, x):
        pos = self.encoder(g, x)
        neg = self.encoder(g, x, corrupt=True)
        return pos, neg

    def embed(self, g, x, e=None):
        return self.encoder(g, x)

# Build GraphCL model
def build_model(args):
    in_dim = args.n_dim
    hidden_dim = args.num_hidden
    edge_dim = args.e_dim

    encoder = GCNEncoder(in_dim, hidden_dim, edge_dim)
    model = GraphCL(encoder, hidden_dim)
    return model

# Build hyperparameter-tunable GraphCL model
def build_model_hyper(args):
    in_dim = args.n_dim
    hidden_dim = args.num_hidden
    edge_dim = args.e_dim
    num_layers = args.num_layers

    encoder = GCNEncoder_hyper(in_dim, hidden_dim, edge_dim, num_layers=num_layers)
    model = GraphCL(encoder, hidden_dim)
    return model

# Build DGI model
def build_dgi_model(args):
    in_dim = args.n_dim
    hidden_dim = args.num_hidden

    encoder = DGIEncoder(in_dim, hidden_dim)
    model = DGIModule(encoder, hidden_dim)
    return model

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def embed(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def build_graphsage_model(args):
    in_feats = args.n_dim
    h_feats = args.num_hidden
    num_classes = args.num_classes if hasattr(args, 'num_classes') else h_feats
    model = GraphSAGE(in_feats, h_feats, num_classes)
    return model
