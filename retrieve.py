import torch
import numpy as np
from torch_geometric.data.data import Data
import pandas as pd
import networkx as nx

from lm_modeling import load_model, load_text2embedding
import time
import os
from pcst_fast import pcst_fast


model_name = 'sbert'


class GraphRetrieval:
    def __init__(self, model_name='sbert', path='dataset/fb'):
        self.model_name = model_name
        self.path = path
        self.movie_path = "dataset/ML1M"
        self.model, self.tokenizer, self.device = self.load_model()
        self.text2embedding = self.load_text2embedding()
        #if we set index GNN layer = 3
        self.G = self.load_graph()
        self.G1 = self.load_second_layer_graph()
        self.Graph = self.load_graph_data()
        self.sorted_item_ids=self.sort_item_list()
        self.movie_id_to_name = self.load_name()
   

    def retrieval_topk(self, input, retrieve_movies_list, topk_nodes=3, topk_rerank_nodes=5):
        first_order_retrieved_graphs = []
        second_order_retrieved_graphs = []
        first_order_reranking_subgraph = []
        second_order_reranking_subgraph = []
        graph = []
        for item in retrieve_movies_list:
                    item_name = self.movie_id_to_name[item]
                    q_emb = self.encode_query(item_name)
                    first_order_retrieved_nodes = retrieval_topk(self.G, q_emb, topk_nodes, 0)
                    second_order_retrieved_nodes = retrieval_topk(self.G1, q_emb, topk_nodes, 0)
                    first_order_retrieved_graphs.extend(first_order_retrieved_nodes)
                    second_order_retrieved_graphs.extend(second_order_retrieved_nodes)
        global_q_emb = self.encode_query(input)
        re_ranking_graphs_first_order = self.re_ranking(global_q_emb, first_order_retrieved_graphs, topk_nodes=topk_rerank_nodes)
        re_ranking_graphs_second_order = self.re_ranking_seconde_order(global_q_emb, second_order_retrieved_graphs, topk_nodes=topk_rerank_nodes)
        for nodes in re_ranking_graphs_first_order:
            first_order_subgraph = self.get_first_order_subgraph(nodes)
            first_order_reranking_subgraph.append(first_order_subgraph)
        for nodes in re_ranking_graphs_second_order:
            second_order_subgraph = self.get_first_order_subgraph(nodes)
            second_order_reranking_subgraph.append(second_order_subgraph)
        graph.extend(first_order_reranking_subgraph)
        graph.extend(second_order_reranking_subgraph)

        return graph

    def retrieval_topk_graphlayer(self, q_emb, topk_nodes=5, topk_edges=0):
        return retrieval_topk(self.G1, q_emb, topk_nodes, topk_nodes)

    def encode_query(self, query):
        return encode_query(query, self.model, self.tokenizer, self.device, self.text2embedding)

    def whether_retrieval(self, watching_list, k):
        retrieve_list=[]
        retrieve_list = sorted(watching_list, key=lambda x: self.sorted_item_ids.index(x))
        index_num=int(k)

        return retrieve_list[:index_num]

    def sort_item_list(self):

        input_filename = f'{self.movie_path}/ratings_45.txt'
        item_interaction_count = {}
        with open(input_filename, 'r') as infile:
            for line in infile:
                item_id = line.strip().split('\t')[1]
                item_interaction_count[item_id] = item_interaction_count.get(item_id, 0) + 1


        sorted_items_by_count = sorted(item_interaction_count.items(), key=lambda x: x[1])
        sorted_item_ids = [int(item[0]) for item in sorted_items_by_count]
        return sorted_item_ids

    def get_first_order_subgraph_edge(self, node):
        neighbors = set(self.Graph.neighbors(node))
        subgraph_nodes = {node}.union(neighbors)
        subgraph_edges = [(u, v) for u, v in self.Graph.edges(node)]
        subgraph = nx.Graph()
        subgraph.add_node(node)
        subgraph.add_edges_from(subgraph_edges)

        nodes, edges = self.load_text_data()    
        edge_to_index = {}
        for i, (source, target) in enumerate(self.G.edge_index.t().tolist()):
            edge_to_index[(source, target)] = i

        selected_edges = list(subgraph.edges())
        selected_edge_indices = [edge_to_index[edge] for edge in selected_edges if edge in edge_to_index]  
        selected_edge_attrs = edges.loc[selected_edge_indices, ['src','edge_attr','dst']].values.tolist()
        edge_attrs_list=[]
        for node_1,edge_attrs,node_2 in selected_edge_attrs:
            edge_attrs_list.append([nodes.loc[node_1, 'node_attr'], edge_attrs, nodes.loc[node_2, 'node_attr']])

        return edge_attrs_list

    def get_first_order_subgraph(self, node):
        neighbors = set(self.Graph.neighbors(node))
        subgraph_nodes = {node}.union(neighbors)
        subgraph_edges = [(node, neighbor) for neighbor in neighbors]

        edge_to_index = {}
        for i, (source, target) in enumerate(self.G.edge_index.t().tolist()):
            edge_to_index[(source, target)] = i
            edge_to_index[(target, source)] = i

        node_to_subgraph_index = {old: new for new, old in enumerate(subgraph_nodes)}

        subgraph_edge_index = torch.tensor([[node_to_subgraph_index[src], node_to_subgraph_index[dst]] for src, dst in subgraph_edges]).t().contiguous()

        subgraph_node_features = self.G.x[list(subgraph_nodes)]

        subgraph_edge_features = self.G.edge_attr[[edge_to_index[(src, dst)] for src, dst in subgraph_edges if (src, dst) in edge_to_index]]

        subgraph = Data(x=subgraph_node_features, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_features, num_nodes=len(subgraph_nodes))


        return subgraph

    def get_second_order_subgraph(self, node):
        first_order_neighbors = set(self.Graph.neighbors(node))
        second_order_neighbors = set()
        for neighbor in first_order_neighbors:
            second_order_neighbors.update(self.Graph.neighbors(neighbor))
        second_order_neighbors.discard(node)

        subgraph_nodes = {node}.union(first_order_neighbors).union(second_order_neighbors)

        subgraph_edges = set()
        for neighbor in first_order_neighbors:
            subgraph_edges.add((node, neighbor))
            for second_order_neighbor in self.Graph.neighbors(neighbor):
                if second_order_neighbor in subgraph_nodes:
                    subgraph_edges.add((neighbor, second_order_neighbor))

        edge_to_index = {}
        for i, (source, target) in enumerate(self.G.edge_index.t().tolist()):
            edge_to_index[(source, target)] = i
            edge_to_index[(target, source)] = i 

        node_to_subgraph_index = {old: new for new, old in enumerate(subgraph_nodes)}

        subgraph_edge_index = torch.tensor([[node_to_subgraph_index[src], node_to_subgraph_index[dst]] for src, dst in subgraph_edges]).t().contiguous()

        subgraph_node_features = self.G.x[list(subgraph_nodes)]

        subgraph_edge_features = self.G.edge_attr[[edge_to_index[(src, dst)] for src, dst in subgraph_edges if (src, dst) in edge_to_index]]

        subgraph = Data(x=subgraph_node_features, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_features, num_nodes=len(subgraph_nodes))

        return subgraph
    
    def load_model(self):
        model, tokenizer, device = load_model[model_name]()
        return model, tokenizer, device
    
    def load_text2embedding(self):
        text2embedding = load_text2embedding[model_name]
        return text2embedding

    def load_graph(self):
        G = torch.load(f'{self.path}/graphs/0.pt')
        return G
    
    def load_second_layer_graph(self):
        G_2 = torch.load(f'{self.path}/graphs/layer2_embeddings_W.pt')
        return G_2
    
    def load_graph_data(self):
        Graph = nx.Graph()
        Graph.add_nodes_from(range(self.G.num_nodes))
        Graph.add_edges_from(self.G.edge_index.t().tolist())
        return Graph
    
    def load_name(self):
        moviesidnamepath = f'{self.movie_path}/movies_id_name.txt'
        movie_id_to_name = {}
        with open(moviesidnamepath, 'r') as file:
            for line in file:
                movieid, moviename = line.strip().split('\t')
                movie_id_to_name[int(movieid)] = moviename
        return movie_id_to_name


    def re_ranking(self, q_emb, retrieved_graphs, topk_nodes=5):
        features_list = [self.G.x[node] for node in retrieved_graphs]
        features_tensor = torch.stack(features_list)
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, features_tensor)
        _, topk_n_indices = torch.topk(n_prizes.squeeze(), topk_nodes, largest=True)
        reranked_nodes = [retrieved_graphs[i] for i in topk_n_indices]

        return reranked_nodes
    
    def re_ranking_seconde_order(self, q_emb, retrieved_graphs, topk_nodes=5):
        features_list = [self.G1.x[node] for node in retrieved_graphs]
        features_tensor = torch.stack(features_list)
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, features_tensor)
        _, topk_n_indices = torch.topk(n_prizes.squeeze(), topk_nodes, largest=True)
        reranked_nodes = [retrieved_graphs[i] for i in topk_n_indices]

        return reranked_nodes



    def load_text_data(self):
        return load_text_data(self.path)
    
    def retrieval_node_texts(self, selected_nodes):
        nodes, edges = self.load_text_data()
        selected_node_texts = nodes.loc[selected_nodes, 'node_attr'].tolist()
        return selected_node_texts
    



def retrieval_topk(graph, q_emb, topk_nodes=1, topk_edges=0):    
    n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
    topk_nodes = min(topk_nodes, graph.num_nodes)
    _, topk_n_indices = torch.topk(n_prizes, topk_nodes, largest=True)
    selected_nodes = topk_n_indices.tolist()
    
    return selected_nodes


def encode_query(query, model, tokenizer, device, text2embedding):
    # print('Encoding query...')
    x = text2embedding(model, tokenizer, device, query)
    return x




def load_text_data(path):
    nodes = pd.read_csv(f'{path}/nodes/all_nodes.csv')
    edges = pd.read_csv(f'{path}/edges/all_edges.csv')
    return nodes, edges
