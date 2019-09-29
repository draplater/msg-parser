import pickle
import random
import time
from operator import itemgetter

import dynet as dn

import graph_utils
import nn
import sys
from common_utils import split_to_batches
from edge_eval_network import EdgeEvaluationNetwork
from vocab_utils import Statistics
from logger import logger
from max_sub_graph import graph_decoders
from max_sub_graph.cost_augment import cost_augmentors
from parser_base import GraphParserBase


class PrintLogger(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss_value = 0.0
        self.total_gold_edge = sys.float_info.epsilon
        self.total_predict_edge = sys.float_info.epsilon
        self.recalled_gold_edge = 0.0
        self.correct_predict_edge = 0.0
        self.start = time.time()

    def print(self, sentence_idx):
        logger.info(
            'Processing sentence number: %d, Loss: %.2f, '
            'Accuracy: %.2f, Recall: %.2f, Time: %.2f',
            sentence_idx, self.total_loss_value,
            self.correct_predict_edge / self.total_predict_edge * 100,
            self.recalled_gold_edge / self.total_gold_edge * 100,
            time.time() - self.start
        )
        self.reset()


class MaxSubGraphParser(GraphParserBase):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        super(MaxSubGraphParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--trainer", type=str, dest="trainer", default="adam", choices=nn.trainers.keys())
        group.add_argument("--cost-augment", type=str, dest="cost_augment", default="hamming", choices=cost_augmentors)
        group.add_argument("--decoder", type=str, dest="decoder", default="arcfactor", choices=graph_decoders.keys())
        group.add_argument("--predict-decoder", type=str, dest="test_decoder", default=None)
        group.add_argument("--hamming-a", type=float, dest="hamming_a", default=0.4)
        group.add_argument("--hamming-b", type=float, dest="hamming_b", default=0.6)
        group.add_argument("--vine-arc-length", type=int, dest="vine_arc_length", default=20)
        group.add_argument("--basic-costaug-decrease", type=int, dest="basic_costaug_decrease", default=1)
        group.add_argument("--loose-value", type=float, dest="loose", default=-1)
        group.add_argument("--delta", type=float, dest="delta", default=1)

        EdgeEvaluationNetwork.add_parser_arguments(arg_parser)

    @classmethod
    def add_common_arguments(cls, arg_parser):
        super(MaxSubGraphParser, cls).add_common_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__ + " (common)")
        group.add_argument("--batch-size", type=int, dest="batch_size", default=32)

    def __init__(self, options, train_graphs=None, restore_model_and_saveable=None, train_extra=None):
        random.seed(1)

        self.decoder = graph_decoders[options.decoder](options)
        self.test_decoder = graph_decoders[options.test_decoder](options) \
            if options.test_decoder is not None \
            else self.decoder
        self.cost_augment = cost_augmentors[options.cost_augment](options)

        self.options = options
        if "func" in options:
            del options.func

        self.labelsFlag = options.labelsFlag

        if restore_model_and_saveable:
            self.model, self.network = restore_model_and_saveable
        else:
            self.model = dn.Model()
            self.trainer = nn.trainers[options.trainer](self.model)
            if train_extra is not None:
                statistics = Statistics.from_sentences(train_graphs + train_extra)
            else:
                statistics = Statistics.from_sentences(train_graphs)
            self.network = EdgeEvaluationNetwork(self.model, statistics, options)

    def predict_session(self, sentence):
        """
        step 1: yield all edge expressions
        step 2: yield all label expressions
        step 3: yield result graph
        """
        lstm_output = self.network.get_lstm_output(sentence)
        length = len(sentence)
        raw_exprs = self.network.edge_eval.get_complete_raw_exprs(lstm_output)
        yield raw_exprs

        scores = self.network.edge_eval.raw_exprs_to_scores(raw_exprs, length)
        output_graph = self.test_decoder(scores)

        edges = []
        for source_id in range(len(sentence)):
            for target_id in range(len(sentence)):
                if target_id == 0:  # avoid edges pointed to root
                    continue
                if output_graph[source_id][target_id]:
                    edges.append(graph_utils.Edge(source_id, "X", target_id))

        if self.labelsFlag:
            labeled_edges = []
            labels_exprs = list(self.network.label_eval.get_label_scores(lstm_output, edges))
            yield labels_exprs
            for edge, r_scores_expr in zip(edges, labels_exprs):
                r_scores = r_scores_expr.value()
                label_index = max(((l, scr) for l, scr in enumerate(r_scores)), key=itemgetter(1))[0]
                label = self.network.irels[label_index]
                labeled_edges.append(graph_utils.Edge(edge.source, label, edge.target))
            edges = labeled_edges
        else:
            yield []

        result = sentence.replaced_edges(edges)
        yield result

    def predict(self, graphs):
        self.network.sent_embedding.rnn.disable_dropout()
        for sentence_idx, batch_idx, batch_sentences in split_to_batches(
                graphs, self.options.batch_size):
            sessions = [self.predict_session(sentence)
                        for sentence in batch_sentences]
            all_exprs = [next(i) for i in sessions]
            if all_exprs:
                dn.forward(all_exprs)
            all_labels_exprs = [j for i in sessions for j in next(i)]
            if all_labels_exprs:
                dn.forward(all_labels_exprs)
            for i in sessions:
                yield next(i)
            dn.renew_cg()


    def training_session(self, sentence, print_logger, loose_var=-1):
        """
        step 1: yield all edge expressions
        step 2: yield all label expressions
        step 3: yield loss
        """
        lstm_output = self.network.get_lstm_output(sentence)
        length = len(sentence)
        raw_exprs = self.network.edge_eval.get_complete_raw_exprs(lstm_output)
        yield raw_exprs

        scores = self.network.edge_eval.raw_exprs_to_scores(raw_exprs, length)
        exprs = self.network.edge_eval.raw_exprs_to_exprs(raw_exprs, length)

        self.cost_augment(scores, sentence)

        output_graph = self.decoder(scores)
        gold_graph = sentence.to_matrix()

        label_loss = dn.scalarInput(0.0)
        if self.labelsFlag:
            edges = list(sentence.generate_edges())
            labels_exprs = list(self.network.label_eval.get_label_scores(lstm_output, edges))
            yield labels_exprs
            for edge, r_scores_expr \
                    in zip(edges, labels_exprs):
                head, label, modifier = edge
                r_scores = r_scores_expr.value()
                gold_label_index = self.network.rels[label]
                wrong_label_index = max(((l, scr)
                                         for l, scr in enumerate(r_scores)
                                         if l != gold_label_index), key=itemgetter(1))[0]
                # if loose_var is set, we could do something to loose the update of tagging
                delta = self.options.delta if loose_var > 0 else 1
                if r_scores[gold_label_index] < r_scores[wrong_label_index] + delta:
                    label_loss += r_scores_expr[wrong_label_index] - r_scores_expr[gold_label_index] + 1
        else:
            yield []

        edge_loss = dn.scalarInput(0.0)
        for source_id in range(len(sentence)):
            for target_id in range(len(sentence)):
                gold_exist = gold_graph[source_id][target_id]
                output_exist = output_graph[source_id][target_id]
                if gold_exist and output_exist:
                    print_logger.total_gold_edge += 1
                    print_logger.total_predict_edge += 1
                    print_logger.correct_predict_edge += 1
                    print_logger.recalled_gold_edge += 1
                elif not gold_exist and not output_exist:
                    pass
                elif gold_exist and not output_exist:
                    print_logger.total_gold_edge += 1
                    if loose_var > 0 and scores[source_id][target_id] > -loose_var: #-0.1:
                        pass
                    else:
                        edge_loss -= exprs[source_id][target_id]
                elif not gold_exist and output_exist:
                    print_logger.total_predict_edge += 1
                    if loose_var > 0 and scores[source_id][target_id] < loose_var: #0.05:
                        pass
                    else:
                        edge_loss += exprs[source_id][target_id]
                else:
                    raise SystemError()

        loss_shift = self.cost_augment.get_loss_shift(output_graph, gold_graph)
        loss = label_loss + edge_loss + loss_shift
        yield loss

    def train_gen(self, graphs, update=True, extra=None):
        """
        :type graphs: list[graph_utils.Graph]
        """
        self.logger = PrintLogger()
        self.network.sent_embedding.rnn.set_dropout(self.options.lstm_dropout)
        print_per = (100 // self.options.batch_size + 1) * self.options.batch_size

        if extra is not None:
            for sentence_idx, batch_idx, batch_sentences in split_to_batches(
                    extra, self.options.batch_size):
                if sentence_idx % print_per == 0 and sentence_idx != 0:
                    self.logger.print(sentence_idx)
                sessions = [self.training_session(sentence, self.logger, loose_var=self.options.loose)
                            for sentence in batch_sentences]
                all_exprs = [next(i) for i in sessions]
                if all_exprs:
                    dn.forward(all_exprs)
                all_labels_exprs = [j for i in sessions for j in next(i)]
                if all_labels_exprs:
                    dn.forward(all_labels_exprs)
                loss = sum(next(i) for i in sessions) / len(sessions)
                self.logger.total_loss_value += loss.value()
                if update:
                    loss.backward()
                    self.trainer.update()
                    dn.renew_cg()
                    sessions.clear()

        for sentence_idx, batch_idx, batch_sentences in split_to_batches(
                graphs, self.options.batch_size):
            if sentence_idx % print_per == 0 and sentence_idx != 0:
                self.logger.print(sentence_idx)
            sessions = [self.training_session(sentence, self.logger)
                        for sentence in batch_sentences]
            all_exprs = [next(i) for i in sessions]
            if all_exprs:
                dn.forward(all_exprs)
            all_labels_exprs = [j for i in sessions for j in next(i)]
            if all_labels_exprs:
                dn.forward(all_labels_exprs)
            loss = sum(next(i) for i in sessions) / len(sessions)
            self.logger.total_loss_value += loss.value()
            if update:
                loss.backward()
                self.trainer.update()
                dn.renew_cg()
                sessions.clear()
            yield (loss if not update else None)

    def train(self, graphs, extra=None):
        for _ in self.train_gen(graphs, extra=extra):
            pass

    def save(self, prefix):
        nn.model_save_helper("pickle", prefix, self.network, self.options)

    @classmethod
    def load(cls, prefix, new_options=None):
        """
        :param prefix: model file name prefix
        :type prefix: str
        :rtype: MaxSubGraphParser
        """
        model = dn.Model()
        options, savable = nn.model_load_helper(None, prefix, model)
        options.__dict__.update(new_options.__dict__)
        ret = cls(options, None, (model, savable))
        return ret
