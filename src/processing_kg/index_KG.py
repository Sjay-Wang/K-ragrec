import re
import os
import torch
import pandas as pd

from tqdm import tqdm
from torch_geometric.data.data import Data
from lm_modeling import load_model, load_text2embedding

import torch
from torch.nn import Linear
from torch.nn.functional import normalize
import torch.nn.functional as F


model_name = 'sbert'
path = 'dataset/fb'
dataset = pd.read_csv(f'{path}/mapped_filtered_fb.txt', sep='\t')



triplets = []
with open('dataset/fb/mapped_filtered_fb.txt', 'r') as file:
    for line in file:
        triplets.append(line.strip())


    
def textualize_graph(triplets_list, node_ids):
    nodes = []
    edges = []
    for triplet in triplets_list:
        subject, predicate, obj = triplet.split('\t')
        if subject not in node_ids:
            node_ids[subject] = len(node_ids)
        if obj not in node_ids:
            node_ids[obj] = len(node_ids)
        nodes.append({'node_id': node_ids[subject], 'node_attr': subject})
        nodes.append({'node_id': node_ids[obj], 'node_attr': obj})
        edges.append({'src': node_ids[subject], 'edge_attr': predicate, 'dst': node_ids[obj]})
    return pd.DataFrame(nodes), pd.DataFrame(edges)

def step_one():
    os.makedirs(f'{path}/nodes', exist_ok=True)
    os.makedirs(f'{path}/edges', exist_ok=True)
    all_nodes = pd.DataFrame()
    all_edges = pd.DataFrame()
    node_ids = {} 

    for i, triplet_str in enumerate(tqdm(triplets, total=len(triplets))):
        nodes, edges = textualize_graph([triplet_str], node_ids)
        all_nodes = pd.concat([all_nodes, nodes], ignore_index=True)
        all_edges = pd.concat([all_edges, edges], ignore_index=True)

    all_nodes = all_nodes.drop_duplicates(subset=['node_id'])
    all_nodes.to_csv(f'{path}/nodes/all_nodes.csv', index=False)
    all_edges.to_csv(f'{path}/edges/all_edges.csv', index=False)


def step_two():

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        nodes = pd.read_csv(f'{path}/nodes/all_nodes.csv')
        edges = pd.read_csv(f'{path}/edges/all_edges.csv')
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src, edges.dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
        torch.save(data, f'{path}/graphs/0.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_graph()

    
class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=x.dtype)
        deg = deg.scatter_add_(0, row, torch.ones_like(row, dtype=x.dtype))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

        x = self.linear(x)
        out = torch.sparse.mm(adj, x)

        return out
    
class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 1024)            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        e, nodes_num = data.edge_attr, data.num_nodes

        x = F.relu(self.conv1(x, edge_index))
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=nodes_num)
        torch.save(data, f'{path}/graphs/layer2_embeddings_W.pt')

        x = F.relu(self.conv2(x, edge_index))
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=nodes_num)
        torch.save(data, f'{path}/graphs/layer3_embeddings_W.pt')


        return x



if __name__ == '__main__':
    if os.path.exists(f'{path}/nodes/all_nodes.csv') and os.path.exists(f'{path}/edges/all_edges.csv'):
        print("Files has exist.")
    else:
        step_one()
    step_two()
    data = torch.load(f'{path}/graphs/0.pt', weights_only=False)
    model = GCN(num_features=data.x.shape[1])
    model.eval()
    model(data)