from grm.util import merge_dict, avg_dict, filter_log_by_caseid, get_values_event_attribute_for_trace, create_pig, \
    log_to_numeric, __getmappingactivities__, __getmappinglabel__, norm_min_max_dict
from grm.GGNNsparse import GGNNsparse
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory
from pm4py.algo.filtering.log.attributes import attributes_filter
from grm.preprocessing import preprocess
import eval.util.metrics as metrics
import csv


class GRM(GGNNsparse):
    def __init__(self, input_data, activities, params=None, restore_file=None, freeze_graph_model=False):
        """
        GRM: Graph relevance miner.
        An extension of the sparse GGNN model for process outcome prediction
        (i.e. classification) in Predictive Business Process Monitoring.
        :param input_data: event log as a list of traces with a label [list]
        :param params: hyperparameters of the model [dict]
        :param restore_file: path to a model that should be restored [str]
        :param freeze_graph_model: do not train parameters of graph model (i.e. model is not trained) [bool]
        """

        """ 
        Preprocess data: 
        - transform to numerical
        - create a mapping table
        - split data into training
        - testing and evaluation
        """

        self.mapping, data = preprocess(input_data, activities)  # mapping = {list of cases, list of activities, list of labels}
        training, validation = data


        # Get number of different activities which is required for building the sparsity matrix
        num_different_activities = len(self.mapping["activities"])

        # Transform cases into process instance graphs (IGs)
        training_data = create_pig(training, num_different_activities)
        validation_data = create_pig(validation, num_different_activities)

        super().__init__(training_data, validation_data, params=params, restore_file=restore_file,
                         freeze_graph_model=freeze_graph_model)
        print("New GRM model created")

    @classmethod
    def default_params(cls):
        """
        Gets default parameters.
        :return: params [dict].
        """

        params = dict(super().default_params())
        return params

    def train(self):
        """
        Performs training.
        :return: none.
        """

        super().train()

    def predict(self, trace):
        """
        Performs prediction for a trace.
        :param trace: list of activities [list].
        :return: none.
        """

        # Get number of different activities which is required for building the sparsity matrix
        num_different_activities = len(self.mapping["activities"])

        # Transform cases into process instance graphs (IGs)
        trace_in = log_to_numeric([trace], self.mapping)
        pigs = create_pig(trace_in, num_different_activities)

        # Use model for prediction
        eval_out = self.evaluate(pigs)
        result = eval_out[0]

        # Translate numeric prediction value to text label
        pred = self.mapping["labels"][result[0]]

        # Add relevance scores to respective activity label
        relevance_scores = {}
        different_activities_trace = get_values_event_attribute_for_trace(trace, "concept:name")
        for activity in different_activities_trace:
            activity_index = __getmappingactivities__(self.mapping, activity)
            relevance_scores[activity] = result[1][activity_index]
        case_id = trace.attributes["concept:name"]

        return [case_id, pred, relevance_scores]

    def aggregate_relevance_scores(self, log):
        """
        Aggregates relevance scores over the complete event log.
        :param log: event log as a list of traces [list].
        :return: aggregated relevance scores [dict].
        """

        label_0 = self.mapping["labels"][0]
        label_1 = self.mapping["labels"][1]

        aggregated_relevance_scores = {}
        aggregated_relevance_scores[label_0] = {}
        aggregated_relevance_scores[label_1] = {}
        predictions = []
        traces = {}
        traces[label_0] = []
        traces[label_1] = []

        # Aggregate relevance scores through prediction
        for trace in log:
            predictions.append(self.predict(trace))  # a prediction = [case_id, pred, relevance_score]
        for prediction in predictions:
            aggregated_relevance_scores[prediction[1]] = merge_dict(
                aggregated_relevance_scores[prediction[1]],
                prediction[2])
            traces[prediction[1]].append(prediction[0])

        # Averaged relevance scores
        avg_relevance_scores = {}
        avg_relevance_scores[label_0] = {}
        avg_relevance_scores[label_1] = {}
        for key in aggregated_relevance_scores:
            avg_relevance_scores[key]['scores'] = norm_min_max_dict(avg_dict(aggregated_relevance_scores[key]))
            avg_relevance_scores[key]['traces'] = traces[key]
            if len(traces[key]) == 0:
                del avg_relevance_scores[key]

        return avg_relevance_scores

    def testing_log(self, log, verbose=False):
        """
        Perform testing.
        :param log: event log as a list of traces [list]
        :param verbose: print output or not [bool]
        :return: metrics [dict]
        """

        predictions = list()
        labels = list()

        # Collect labels and predictions
        for trace in log:
            prediction = self.predict(trace)
            if verbose is True:
                print("Prediction for instance: " + str(prediction[0]))
                print("Prediction: " + str(prediction[1]))
                print("Relevance distribution")
                print(prediction[2])
            for event in trace:
                labels.append(__getmappinglabel__(self.mapping, event.__getitem__("label")))
                predictions.append(self.bool_to_bin(prediction[1]))
                break

        # Save labels and predictions
        with open('../eval/results/gt_pred.csv', 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['Label', 'Prediction']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for label, prediction in zip(labels, predictions):

                writer.writerow({'Label': label, 'Prediction': prediction})

        return metrics.metrics_from_prediction_and_label(labels, predictions)

    def bool_to_bin(self, bool_value):
        """
        Helper function to map a boolean value to a binary value.
        :param bool_value: boolean boolean value [bool]
        :return: binary value [int]
        """
        if bool_value:
            return 1
        else:
            return 0

    def visualize_dfg(self, log, save_file=False, file_name="dfg", variant="relevance"):
        """
        Visualises the event log as direct follower graph (DFG).
        :param log: event log as a list of traces [list].
        :param save_file: boolean flog indicating to save the DFG or not [bool].
        :param file_name: name of the file [str].
        :param variant: dfg version to be produced: "frequency", "time", "relevance" or "all" [str]
        :return: file_names [list].
        """
        parameters = {"format": "svg"}
        file_names = []
        relevance_scores = self.aggregate_relevance_scores(log)
        if variant == "relevance" or variant == "all":
            for label, items in relevance_scores.items():
                data = filter_log_by_caseid(log, items['traces'])
                dfg = dfg_factory.apply(data)
                gviz = dfg_vis_factory.apply(dfg, activities_count=items['scores'], parameters=parameters)
                if len(items['traces']) == 1:
                    title = "Prediction: " + str(label) + ", Case ID: " + items['traces'][0]
                else:
                    title = "No of Service Orders: " + str(len(log)) + ", Filter: Repair not on time (Label = " + str(label) + ")"
                gviz.body.append('\t// title')
                gviz.body.append('\tfontsize = 50;')
                gviz.body.append('\tlabelloc = "t";')
                gviz.body.append('\tlabel = "' + title + '";')
                print("rel_sc: ", items['scores'])
                if save_file:
                    filen = file_name + "_rel_" + str(label) + ".svg"
                    dfg_vis_factory.save(gviz, filen)
                    print("Saved DFG image to: " + filen)
                    file_names.append(filen)
        if variant == "frequency" or variant == "all":
            for label, items in relevance_scores.items():
                data = filter_log_by_caseid(log, items['traces'])
                dfg = dfg_factory.apply(data)
                activities_cnt = attributes_filter.get_attribute_values(log, attribute_key="concept:name")
                gviz = dfg_vis_factory.apply(dfg, activities_count=activities_cnt, parameters=parameters)
                if len(items['traces']) == 1:
                    title = "Prediction: " + str(label) + ", Case ID: " + items['traces'][0]
                else:
                    title = "No of Service Orders: " + str(len(log)) + ", Filter: Repair not on time (Label = " + str(label) + ")"
                gviz.body.append('\t// title')
                gviz.body.append('\tfontsize = 50;')
                gviz.body.append('\tlabelloc = "t";')
                gviz.body.append('\tlabel = "' + title + '";')
                if save_file:
                    filen = file_name + "_freq_" + str(label) + ".svg"
                    dfg_vis_factory.save(gviz, filen)
                    print("Saved DFG image to: " + filen)
                    file_names.append(filen)
        if variant == "time" or variant == "all":
            for label, items in relevance_scores.items():
                data = filter_log_by_caseid(log, items['traces'])
                dfg = dfg_factory.apply(data)
                parameters = {"format": "svg", "AGGREGATION_MEASURE": "mean"}
                gviz = dfg_vis_factory.apply(dfg, variant="performance", parameters=parameters)
                if len(items['traces']) == 1:
                    title = "Prediction: " + str(label) + ", Case ID: " + items['traces'][0]
                else:
                    title = "No of Service Orders: " + str(len(log)) + ", Filter: Repair not on time (Label = " + str(label) + ")"
                gviz.body.append('\t// title')
                gviz.body.append('\tfontsize = 50;')
                gviz.body.append('\tlabelloc = "t";')
                gviz.body.append('\tlabel = "' + title + '";')
                if save_file:
                    filen = file_name + "_time_" + str(label) + ".svg"
                    dfg_vis_factory.save(gviz, filen)
                    print("Saved DFG image to: " + filen)
                    file_names.append(filen)
            return file_names
