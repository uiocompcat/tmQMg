import os
import torch
import pandas as pd
import scipy
import numpy as np
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Dataset

from tools import Tools
from HyDGL import Graph

class tmQMg(Dataset):

    def __init__(self, root: str, graph_type: str, targets: list[str], exclude: list[str] = []):

        # targets to consider
        self.targets = targets

        # directory to get raw data from
        self.root = root
        self._raw_dir = root + '/' + graph_type +'/'

        # if root path does not exist create folder
        if not os.path.isdir(root):
            os.makedirs(root, exist_ok=True)

        # check if graph data is present
        if not os.path.isdir(self._raw_dir):
            self.download()

        #
        self._file_paths = [self._raw_dir + '/' + file_name for file_name in os.listdir(self._raw_dir)] 

        # list of files to exclude
        self._files_to_exclude = exclude

        # start super class
        super().__init__(self.root)

    @property
    def file_paths(self):
        return self._file_paths

    @property
    def raw_dir(self):
        return self._raw_dir

    @property
    def raw_file_names(self):
        return (pd.read_csv('../data/tmQMg_properties_and_targets.csv')['id'] + '.gml').tolist()

    @property
    def processed_file_names(self):
        return 'data'

    def download(self):
        """Function to download raw data."""
        print('Trying to download..')
        raise NotImplementedError('Download function is not implemented.')

    def len(self):
        """Getter for the number of processed pytorch graphs."""
        return len(self.graphs)

    def get(self, idx):
        """Accessor for processed pytorch graphs."""
        return self.graphs[idx]

    def process(self):

        print('Start processing..')

        self.graphs = self.get_pytorch_graphs()

    def get_class_feature_dicts(self):

        """Gets dicts for one-hot enconding class-type features in node and edge features."""

        print('Getting class feature dicts..')

        pivot_graph_object = Graph.from_networkx(nx.read_gml(self.file_paths[0]))

        if len(pivot_graph_object.nodes) == 0:
            node_class_feature_keys = []
        else:
            # get indicies in node feature list that are non-numerical class values
            node_class_feature_keys = Tools.get_class_feature_keys(pivot_graph_object.nodes[0].features)

        if len(pivot_graph_object.edges) == 0:
            edge_class_feature_keys = []
        else:
            # get indicies in edge feature list that are non-numerical class values
            edge_class_feature_keys = Tools.get_class_feature_keys(pivot_graph_object.edges[0].features)

        # class features
        node_class_features = [[] for idx in node_class_feature_keys]
        edge_class_features = [[] for idx in edge_class_feature_keys]

        for file_path in tqdm(self.file_paths):
            # read graph object
            graph_object = Graph.from_networkx(nx.read_gml(file_path))

            # iterate through the nodes of all graphs to obtain lists of class for each of the class indices
            for node in graph_object.nodes:
                for i, key in enumerate(node_class_feature_keys):
                    if node.features[key] not in node_class_features[i]:
                        node_class_features[i].append(node.features[key])

            # iterate through the edges of all graphs to obtain lists of class for each of the class indices
            for edge in graph_object.edges:
                for i, key in enumerate(edge_class_feature_keys):
                    if edge.features[key] not in edge_class_features[i]:
                        edge_class_features[i].append(edge.features[key])

        # build dicts that contain feature keys as keys and lists of possible class features as values
        node_class_feature_dict = {}
        for i, key in enumerate(node_class_feature_keys):
            node_class_feature_dict[key] = edge_class_features[i]

        edge_class_feature_dict = {}
        for i, key in enumerate(edge_class_feature_keys):
            edge_class_feature_dict[key] = edge_class_features[i]

        return node_class_feature_dict, edge_class_feature_dict

    def get_pytorch_graphs(self, skip_disconnected=True):

        """Builds pytorch graphs from previously built graph objects."""

        print('Building pytorch graphs..')

        # get dict for one-hot edge encoding of edges
        node_class_feature_dict, edge_class_feature_dict = self.get_class_feature_dicts()

        graphs = []
        for file_path in tqdm(self.file_paths):

            # skip if filename in exclude list
            if file_path.split('/')[-1].replace('.gml','') in self._files_to_exclude:
                continue

            # read graph object
            graph_object = Graph.from_networkx(nx.read_gml(file_path))
            # replace target dict to only contain requested targets
            new_target_dict = {}
            for target_key in graph_object.targets.keys():
                if target_key in self.targets:
                    new_target_dict[target_key] = graph_object.targets[target_key]
            graph_object._targets = new_target_dict

            # get pytorch graph
            graph = graph_object.get_pytorch_data_object(node_class_feature_dict=node_class_feature_dict, edge_class_feature_dict=edge_class_feature_dict)

            if skip_disconnected:
                # build adjacency matrix from edge index
                adjacency_list = graph.edge_index.detach().numpy().T
                adjacency_matrix = np.zeros((graph.num_nodes, graph.num_nodes))
                for edge in adjacency_list:
                    adjacency_matrix[edge[0], edge[1]] = 1
                # check the number of subgraphs
                if scipy.sparse.csgraph.connected_components(adjacency_matrix)[0] != 1:
                    continue

            graphs.append(graph)

        return graphs

    def get_meta_data_dict(self):

        """Returns a dict of graphs meta data.

        Returns:
            dict: A dict containing the meta data for all graphs.
        """

        print('Getting meta data dict..')

        meta_data_dict = {}

        for file_path in tqdm(self.file_paths):

            # read graph object
            meta_data_dict[file_path.split('/')[-1].replace('.gml', '')] = nx.read_gml(file_path).graph['meta_data']

        return meta_data_dict
