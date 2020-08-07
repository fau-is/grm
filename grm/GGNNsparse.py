from typing import Tuple, Dict, Sequence, Any
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
from .GGNN import GGNN
from .util import glorot_init, SMALL_NUMBER

GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells', ])


class GGNNsparse(GGNN):

    def __init__(self, data_training, data_testing, params=None, restore_file=None, freeze_graph_model=False):
        """
        Extension of GGNN class to sparse GGNN model.
        :param data_training: data set of PIGs for training [list]
        :param data_testing: data set of PIGs for validation [list]
        :param params: hyperparameters of the model [dict]
        :param restore_file: path to a model that should be restored [str]
        :param freeze_graph_model: do not train parameters of graph model (i.e model is not trained) [bool]
        """
        super().__init__(data_training, data_testing, params=params, restore_file=restore_file,
                         freeze_graph_model=freeze_graph_model)

    @classmethod
    def default_params(cls):
        """
        Returns default parameters of the GGNN model.
        :return: params [dict]
        """

        params = dict(super().default_params())
        params.update({
            'batch_size': 128,
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                "2": [0],
                "4": [0, 2]
            },
            'layer_timesteps': [2, 2, 1, 2, 1],  # Number of layers & propagation steps per layer
            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        """
        Prepare specific GGNN model.
        :return: none.
        """

        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = GGNNWeights([], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights.edge_weights.append(edge_weights)

                # Note we did not use propagation attention.
                if self.params['use_propagation_attention']:
                    self.gnn_weights.edge_type_attention_weights.append(
                        tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                    name='edge_type_attention_weights_%i' % layer_idx))

                # Note we did not use edge biases.
                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(
                        tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert (activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        """
        Compute final node representations.
        :return: node_states_per_layer [matrix]
        """

        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(self.placeholders['initial_node_representation'])
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]

        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(
                        params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                        ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # List of tensors of messages of shape [E, D]
                        message_source_states = []  # List of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                                self.placeholders['adjacency_lists']):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][
                                                                       edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states,
                                                                 message_target_states)  # Shape [M]
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # The following is "softmaxing" over the incoming messages per node.
                            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
                            # Step (1): Obtain shift constant as max of messages going into a node
                            message_attention_score_max_per_target = tf.unsorted_segment_max(
                                data=message_attention_scores,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
                            message_attention_score_max_per_message = tf.gather(
                                params=message_attention_score_max_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message
                            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob
                            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                                data=message_attention_scores_exped,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            message_attention_normalisation_sum_per_message = tf.gather(
                                params=message_attention_score_sum_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention = message_attention_scores_exped / (
                                        message_attention_normalisation_sum_per_message + SMALL_NUMBER)  # Shape [M]
                            # Step (4): Weigh messages using the attention prob
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # Pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[1]  # Shape [V, D]

        return node_states_per_layer[-1]

    def gated_regression(self, last_h, regression_gate, regression_transform):
        """
        Calculates the output of the GGNN model (cf. readout phase).
        :param last_h: final node representations [vector].
        :param regression_gate [object].
        :param regression_transform [object].
        :return: output [flout].
        """

        # Last_h: [v x h]; v = number of nodes; h = number of states.
        # Concatenate the initial and the final state of the nodes.
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]

        # Calculate the node relevances
        self.relevance = tf.nn.sigmoid(regression_gate(gate_input))

        # Calculate the gated outputs
        gated_outputs = self.relevance * regression_transform(last_h)  # [v x 1]

        # Calculate sum over all nodes per graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]

        # Removes dimensions of size 1 from the shape of a tensor
        output = tf.squeeze(graph_representations)  # [g]

        # For binary classification
        output = tf.nn.sigmoid(output)
        self.output = output

        return output

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        """
        Performs data pre-processing and chunking into mini-batches.
        :param raw_data: list of process instance graphs (IG) [list].
        :param is_training_data: boolean indicator if it is for training [bool].
        :return: processed_graphs [list].
        """

        processed_graphs = []
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "labels": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})

        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        """
        Returns adjacency lists for a graph.
        :param graph: process instance graph (IG) [dict]
        :return: adjacency_lists [list]
        """

        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0

            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """
        Create mini-batches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components.
        :param data: set of graphs [list]
        :param is_training: boolean indicator if it is for training [bool]
        :return: batch_feed_dict [dict of tf.placeholders]
        """

        """"""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                target_task_values = []
                target_task_mask = []
                for target_val in cur_graph['labels']:
                    if target_val is None:  # This is one of the examples we didn't sample
                        target_task_values.append(0.)
                        target_task_mask.append(0.)
                    else:
                        target_task_values.append(target_val)
                        target_task_mask.append(1.)
                batch_target_task_values.append(target_task_values)
                batch_target_task_mask.append(target_task_mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    def evaluate(self, data):
        """
        Performs evaluation.
        :param data: list of process instance graphs (IG) [list]
        :return: outputs of GGNN model [list]
        """

        data = self.process_raw_graphs(data, is_training_data=False)
        fetch_list = [self.output, self.relevance]
        batch_feed_dict = self.make_minibatch_iterator(data, is_training=False)
        output = list()
        for item in batch_feed_dict:
            item[self.placeholders['graph_state_keep_prob']] = 1.0
            item[self.placeholders['edge_weight_dropout_keep_prob']] = 1.0
            item[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            item[self.placeholders['target_values']] = [[]]
            item[self.placeholders['target_mask']] = [[]]

            run_out = self.sess.run(fetch_list, feed_dict=item)
            # binary classification
            result = [0 if run_out[0] < 0.5 else 1, run_out[1].tolist()]
            output.append(result)

        return output
