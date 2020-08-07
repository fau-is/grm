from typing import List, Any, Sequence
from .util import MLP, ThreadedIterator, SMALL_NUMBER
import tensorflow as tf
import numpy as np
import time
import pickle
import os
import shutil


class GGNN(object):

    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 1,
            'patience': 250,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,
            'hidden_size': 200,
            'num_timesteps': 4,
            'use_graph': True,
            'task_ids': [0],
            'random_seed': 0,
        }

    def __init__(self, data_training, data_testing, params=None, restore_file=None, freeze_graph_model=False,
                 log_dir="./logged_models", cleanup=False):
        """
        Basic GGNN class that needs to be extended for use.
        :param data_training: data set of PIGs for training [list].
        :param data_testing: data set of PIGs for validation [list].
        :param params: hyperparameters of the model [dict].
        :param restore_file: path to a model that should be restored [str].
        :param freeze_graph_model: do not train parameters of graph model (i.e. model is not trained) [bool].
        :param log_dir: directory where the model is stored [str].
        :param cleanup: clean directory, where the model is stored, before storing it [bool].
        """

        # Collect parameters
        store_params = params
        self.params = self.default_params()
        if store_params is not None:
            self.params.update(store_params)

        # Load data
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(data_training, is_training_data=True)
        self.valid_data = self.load_data(data_testing, is_training_data=False)
        self.freeze_graph_model = freeze_graph_model
        self.restore = restore_file

        # Safe best models/cleanup previous models
        if cleanup:
            shutil.rmtree(log_dir, ignore_errors=True)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Path to best model
        self.best_model_file = os.path.join(log_dir,
                                            "%s_best_model.pickle" % "_".join([time.strftime("%Y-%m-%d-%H-%M")]))

        # Build the actual GGNN model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        """
        Returns graph string from string.
        :param graph_string: graph as string [str].
        :return: graph string as array [list].
        """

        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def load_data(self, data, is_training_data: bool):
        """
        Loads data
        :param data: list of graphs [list]
            A graph = {targets, graph, node_features}
        :param is_training_data: boolean flag if data is for training or not [bool]
        :return: raw process graphs [list]
            A raw process graph = {adjacency_lists [dict], num_incoming_edge per_type [dict],
                init [list], labels [list]}

        """

        num_fwd_edge_types = 0

        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types)
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        """
        Makes the GGNN model.
        :return: none.
        """

        # Create placeholders for the GGNN model
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')
        # Start message passing phase (i.e. update of node representations)
        with tf.variable_scope("graph_mode"):
            self.prepare_specific_graph_model()
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        # Start readout phase (i.e. mapping of node representations to output
        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):

            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'])
                # Computes the output of the GGNN model
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])

                # Computes the difference
                diff = self.placeholders['target_values'][internal_id, :] - computed_values

                # Ignore none comparisons
                task_target_mask = self.placeholders['target_mask'][internal_id, :]
                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
                diff = diff * task_target_mask  # Mask out unused values

                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.cast(tf.equal(tf.round(computed_values),
                                                                                       self.placeholders[
                                                                                           'target_values'][internal_id,
                                                                                       :]), tf.float32))
                # Calculate loss (here, normalised mean squared error)
                task_loss = tf.reduce_sum(tf.square(diff)) / task_target_num

                # Normalise loss
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)

        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        """
        Performs a training step.
        :return: none.
        """

        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.freeze_graph_model:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdadeltaOptimizer(1.0)
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, data, is_training: bool):
        """
        Performs an epoch (i.e. learning iteration).
        :param data: set of graphs [list].
        :param is_training:  boolean flag if data is for training or not [bool].
        :return: loss [list], accuracies [list], error_ratios [list], instance_per_sec [list].
        """

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]

        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies))

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = 1 - accuracies
        instance_per_sec = processed_graphs / (time.time() - start_time)

        return loss, accuracies, error_ratios, instance_per_sec

    def train(self):
        """
        Train the GGNN model.
        :return: none.
        """

        with self.graph.as_default():
            if self.restore is not None:
                # Epoch resume training
                _, valid_accs, _, _ = self.run_epoch(self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (0, 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)

                # Epoch train
                train_loss, train_acc, train_errs, train_speed = self.run_epoch(self.train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_acc)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                # Epoch validation
                valid_loss, valid_accs, valid_errs, valid_speed = self.run_epoch(self.valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc > best_val_acc:

                    # Save best model to self.best_model_file
                    self.save_model(self.best_model_file)
                    print("LOG:  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" %
                          (val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("LOG: Stopping training after %i epochs without improvement on validation accuracy." %
                          self.params['patience'])
                    break

    def save_model(self, model_path: str) -> None:
        """
        Saves the GGNN model.
        :param model_path: path of GGNN model [str].
        :return: none.
        """

        weights_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_save
            weights_save[variable.name] = self.sess.run(variable)

        model_save = {"params": self.params, "weights": weights_save}

        with open(model_path, 'wb') as out_file:
            pickle.dump(model_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        """
        Initialises the GGNN model.
        :return: none.
        """

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        """
        Restores a GGNN model.
        :param path: path of model [str]
        :return: none.
        """

        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as rest_file:
            data_to_load = pickle.load(rest_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Different task_ids possible
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
