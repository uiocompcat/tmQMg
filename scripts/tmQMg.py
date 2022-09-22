import os
import torch
import scipy
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset

from HyDGL.tools import Tools
from HyDGL.qm_data import QmData
from HyDGL.file_handler import FileHandler
from HyDGL.graph_generator import GraphGenerator
from HyDGL.graph_generator_settings import GraphGeneratorSettings


class tmQMg(Dataset):

    def __init__(self, root: str, raw_dir: str, settings: GraphGeneratorSettings, exclude: list[str] = []):

        # if root path does not exist create folder
        if not os.path.isdir(root):
            os.makedirs(root, exist_ok=True)

        # directory to get raw data from
        self._raw_dir = raw_dir

        # file extensions
        self._raw_file_extension = '.json'
        self._graph_object_file_extension = '.graph'
        self._pytorch_graph_file_extension = '.pt'

        # set dataset file names
        self._files_names = list(filter(None, FileHandler.read_file(self.raw_dir + 'names').split('\n')))
        # list of files to exclude
        self._files_to_exclude = exclude

        # set up graph generator
        self._graph_generator = GraphGenerator(settings)

        # start super class
        super().__init__(root)

    @property
    def file_names(self):
        return self._files_names

    @property
    def raw_dir(self):
        return self._raw_dir

    @property
    def pytorch_geometric_dir(self):
        return self.root + '/pyg'

    @property
    def graph_object_dir(self):
        return self.root + '/graph_objects'

    @property
    def raw_file_names(self):
        return [file_name + self._raw_file_extension for file_name in self.file_names]

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

        print('')

        self.build_graph_objects()

        self.build_pytorch_graphs()

        self.graphs = self.get_built_pytorch_graphs()

    def get_class_feature_dicts(self):

        """Gets dicts for one-hot enconding class-type features in node and edge features."""

        graph_object_files = [file for file in os.listdir(self.graph_object_dir)]

        pivot_graph_object = FileHandler.read_binary_file(self.graph_object_dir + '/' + graph_object_files[0])

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

        for file_name in graph_object_files:
            # read graph object
            graph_object = FileHandler.read_binary_file(self.graph_object_dir + '/' + file_name)

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

    def build_graph_objects(self):

        """Builds graph objects from previously extracted QmData."""

        print('Building graph objects..')

        # create graph_object directory if it does not exist
        if not os.path.isdir(self.graph_object_dir):
            os.mkdir(self.graph_object_dir)

        graph_object_files = [file for file in os.listdir(self.graph_object_dir)]
        for file_name in tqdm(self.file_names):

            # if graph is not built yet
            if file_name + self._graph_object_file_extension not in graph_object_files:
                # read QmData file
                qm_data = QmData.from_dict(FileHandler.read_dict_from_json_file(self.raw_dir + '/' + file_name + self._raw_file_extension))
                # build graph
                graph_object = self._graph_generator.generate_graph(qm_data)
                # write to file
                FileHandler.write_binary_file(self.graph_object_dir + '/' + file_name + self._graph_object_file_extension, graph_object)

    def build_pytorch_graphs(self):

        """Builds pytorch graphs from previously built graph objects."""

        print('Building pytorch graphs..')

        # create pytorch directory if it does not exist
        if not os.path.isdir(self.pytorch_geometric_dir):
            os.mkdir(self.pytorch_geometric_dir)

        # get dict for one-hot edge encoding of edges
        node_class_feature_dict, edge_class_feature_dict = self.get_class_feature_dicts()

        pytorch_graph_files = [file for file in os.listdir(self.pytorch_geometric_dir)]
        for file_name in tqdm(self.file_names):

            if file_name + self._pytorch_graph_file_extension not in pytorch_graph_files:
                # read graph object
                graph_object = FileHandler.read_binary_file(self.graph_object_dir + '/' + file_name + self._graph_object_file_extension)
                # get pytorch graph
                graph = graph_object.get_pytorch_data_object(node_class_feature_dict=node_class_feature_dict, edge_class_feature_dict=edge_class_feature_dict)
                # write to file
                torch.save(graph, self.pytorch_geometric_dir + '/' + file_name + self._pytorch_graph_file_extension)

    def get_built_pytorch_graphs(self, skip_disconnected: bool = True):

        """Loads the pytorch graphs from pytorch directory.

        Returns:
            list[pyg.Data]: A list of pytorch geometric graphs.
        """

        print('Loading pytorch graphs..')

        graphs = []
        for file_name in tqdm(self.file_names):

            # skip if filename in exclude list
            if file_name in self._files_to_exclude:
                continue

            graph = torch.load(self.pytorch_geometric_dir + '/' + file_name + self._pytorch_graph_file_extension)

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

        for file_name in tqdm(self.file_names):

            # read graph object
            graph_object = FileHandler.read_binary_file(self.graph_object_dir + '/' + file_name + self._graph_object_file_extension)
            meta_data_dict[file_name] = graph_object.meta_data

        return meta_data_dict

    def clear_directories(self):

        """Deletes all files in the raw, graph_objects and processed directories."""

        self.clear_raw_dir()
        self.clear_graph_directories()

    def clear_graph_directories(self):

        """Deletes all graph files in the graph_objects and processed directories."""

        self.clear_graph_object_dir()
        self.clear_processed_dir()

    def clear_raw_dir(self):

        """Deletes all files in the raw directory."""

        print('This will delete the files in the raw data directory.')
        reply = input('Do you really wish to continue? [Y/n]')
        while not (reply.strip().lower() == 'y' or reply.strip().lower() == 'n'):
            reply = input('Please anser with "y" or "n".')

        if reply.lower() == 'y':
            FileHandler.clear_directory(self.raw_dir, [file_name + self._raw_file_extension for file_name in self.file_names])
        elif reply.lower() == 'n':
            print('Aborting. If you only wish to clear the directories containing graph representations use the "clear_graph_directories()" function.')

    def clear_graph_object_dir(self):

        """Deletes all files in the graph_objects directory."""

        FileHandler.clear_directory(self.graph_object_dir, [file_name + self._graph_object_file_extension for file_name in self.file_names])

    def clear_processed_dir(self):

        """Deletes all files in the processed directory."""

        FileHandler.clear_directory(self.processed_dir, [file_name + self._pytorch_graph_file_extension for file_name in self.file_names])
