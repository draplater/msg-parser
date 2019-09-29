from __future__ import division

import random

import dynet as dn
import math

import nn

from logger import logger
from vocab_utils import Dictionary, Statistics


class SentenceEmbeddings(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--wembedding", type=int, dest="wembedding_dims", default=100)
        group.add_argument("--pembedding", type=int, dest="pembedding_dims", default=25)
        group.add_argument("--postag-dropout", type=float, dest="postag_dropout", default=0.0)
        group.add_argument("--cembedding", type=int, dest="cembedding_dims", default=100)
        group.add_argument("--supertag-embedding", type=int, dest="supertag_embedding", default=0)
        group.add_argument("--supertag-dropout", type=float, dest="supertag_dropout", default=0.0)
        group.add_argument("--extrn-supertag-embedding", dest="ext_supertag_embedding", metavar="FILE")
        group.add_argument("--word-threshold", type=int, dest="word_threshold", default=0)
        group.add_argument("--word-fallback", type=float, dest="word_fallback", default=0.0)
        group.add_argument("--clstmlayers", type=int, dest="clstm_layers", default=2)
        group.add_argument("--crnn-type", dest="crnn_type", choices=nn.recurrents, default="lstm")
        group.add_argument("--lstmlayers", type=int, dest="lstm_layers", default=2)
        group.add_argument("--highway-layers", type=int, dest="highway_layers", default=0)
        group.add_argument("--lstmdims", type=int, dest="lstm_dims", default=125)
        group.add_argument("--rnn-type", dest="rnn_type", choices=nn.recurrents, default="lstm")
        group.add_argument("--lstm-dropout", type=float, dest="lstm_dropout", default=0.0)
        group.add_argument("--extrn", dest="ext_embedding", help="External embeddings", metavar="FILE")

    def __init__(self, model, statistics, options):
        super(SentenceEmbeddings, self).__init__(model)
        self.options = options

        self.ldims = options.lstm_dims

        if options.ext_embedding is not None:
            self.ext_embedding = nn.ExternalEmbedding(self, options.ext_embedding)
            e_dim = self.ext_embedding.dim
            logger.info('Load external embedding. Vector dimensions %d', self.ext_embedding.dim)
        else:
            self.ext_embedding = None
            e_dim = 0

        self.total_dims = options.wembedding_dims + options.pembedding_dims + options.supertag_embedding + e_dim

        rnn_dims = [self.total_dims] + [self.ldims * 2] * options.lstm_layers
        if self.options.highway_layers <= 0:
            self.rnn = nn.recurrents[options.rnn_type](self, rnn_dims)
        else:
            self.rnn = nn.HighWayRecurrentWrapper(self, rnn_dims,
                                                  self.options.highway_layers,
                                                  nn.recurrent_builders[options.rnn_type])

        if options.cembedding_dims > 0 and options.word_threshold > 1:
            self.char_embedding = nn.Embedding(self, list(statistics.characters), options.cembedding_dims)
            self.c_lstm = nn.recurrents[options.crnn_type](
                self, [options.cembedding_dims] + [options.wembedding_dims] * options.lstm_layers)
            self.freq_words = set(word for word, count in statistics.words.items()
                                  if count >= options.word_threshold)
            logger.info("Word embedding size: {}".format(len(self.freq_words)))
            self.word_embedding = nn.Embedding(self, self.freq_words, options.wembedding_dims)
        else:
            self.word_embedding = nn.Embedding(self, list(statistics.words), options.wembedding_dims)

        if options.pembedding_dims > 0:
            self.pos_embedding = nn.Embedding(self, list(statistics.postags), options.pembedding_dims)
        else:
            self.pos_embedding = None

        if options.supertag_embedding > 0:
            self.supertag_embedding = nn.EmbeddingFromDictionary(
                self, statistics.supertags, options.supertag_embedding,
                external_init=options.ext_supertag_embedding)
        else:
            self.supertag_embedding = None

        self.rel_embedding = nn.Embedding(self, list(statistics.labels), options.pembedding_dims, ())
        self.random = random.Random(168)

    def restore_components(self, components):
        if self.options.ext_embedding is not None:
            self.ext_embedding = components.pop(0)
        self.rnn = components.pop(0)
        if self.options.cembedding_dims > 0 and self.options.word_threshold > 1:
            self.char_embedding = components.pop(0)
            self.c_lstm = components.pop(0)
        self.word_embedding = components.pop(0)

        if self.options.pembedding_dims > 0:
            self.pos_embedding = components.pop(0)
        else:
            self.pos_embedding = None

        if getattr(self.options, "supertag_embedding", 0) > 0:
            self.supertag_embedding = components.pop(0)
        else:
            self.supertag_embedding = None

        self.rel_embedding = components.pop(0)
        assert not components

    def get_vecs(self, node):
        #  word -> input vector of LSTM
        need_word_fallback = hasattr(self, "is_train") and self.options.is_train and \
                             hasattr(self, "word_fallback") and \
                             self.options.word_fallback > 0 and \
                             self.random.random() < self.options.word_fallback
        if not node.norm:
            # empty string
            word_vec = self.word_embedding("*EMPTY*")
        elif self.options.cembedding_dims != 0 and self.options.word_threshold > 1 \
                and (node.norm not in self.freq_words or need_word_fallback):
            # use character vector
            char_vecs = [self.char_embedding(i) for i in node.norm]
            char_vecs_o = self.c_lstm(char_vecs)
            word_vec = (char_vecs_o[0] + char_vecs_o[-1]) / 2
        else:
            # use word vector
            word_vec = self.word_embedding(node.norm)
        vecs = [word_vec]

        if self.options.pembedding_dims > 0:
            if self.options.is_train and self.options.postag_dropout > 0 and \
                    self.random.random() < self.options.postag_dropout:
                pos_vec = self.pos_embedding.lookup[0]
            else:
                pos_vec = self.pos_embedding(node.postag)
            vecs.append(pos_vec)

        if self.options.supertag_embedding > 0:
            if self.options.is_train and self.options.supertag_dropout > 0 and \
                    self.random.random() < self.options.supertag_dropout:
                supertag_vec = self.supertag_embedding.lookup[0]
            else:
                supertag_vec = self.supertag_embedding(node.supertag)
            vecs.append(supertag_vec)

        if self.ext_embedding is not None:
            ext_vec = self.ext_embedding(node.form, (node.norm,))
            vecs.append(ext_vec)

        return dn.concatenate(vecs)

    def get_lstm_output(self, sentence):
        return self.rnn([self.get_vecs(i) for i in sentence])


class EdgeEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--activation", type=str, dest="activation", default="relu")
        group.add_argument("--bilinear-dim", type=int, dest="bilinear_dim", default=100)
        group.add_argument("--mlp-dims", dest="mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[])

    def __init__(self, model, options):
        super(EdgeEvaluation, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]
        self.ldims = options.lstm_dims

        self.bilinear_layer = nn.BiLinear(self, self.ldims * 2, options.bilinear_dim)

        dense_dims = [options.bilinear_dim] + options.mlp_dims + [1]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)

    def restore_components(self, components):
        self.bilinear_layer = components.pop(0)
        self.dense_layer = components.pop(0)
        assert not components

    def get_complete_raw_exprs(self, lstm_output):
        length = len(lstm_output)

        lstm_output_as_batch = dn.concatenate_to_batch(lstm_output)
        headfov = self.bilinear_layer.w1.expr() * lstm_output_as_batch
        modfov = self.bilinear_layer.w2.expr() * lstm_output_as_batch

        # (i, j) -> (i * length + j,)
        # i = k / length, j = k % length
        # 1 1 2 2 3 3 4 4 ..
        heads = [dn.pick_batch_elem(headfov, i) for i in range(length)]
        mods = [dn.pick_batch_elem(modfov, i) for i in range(length)]
        head_part = dn.concatenate_to_batch([heads[i // len(lstm_output)] for i in range(length * length)])
        # 1 2 3 4 .. 1 2 3 4 ...
        mod_part = dn.concatenate_to_batch([mods[i] for i in range(length)] * length)

        output = self.dense_layer(self.activation(head_part + mod_part + self.bilinear_layer.bias.expr()))
        return output

    def raw_exprs_to_exprs(self, output, length=None):
        length = length or int(math.sqrt(output.dims()[0] + 0.5))
        exprs = [[dn.pick_batch_elem(output, i * length + j) for j in range(length)]
                 for i in range(length)]
        return exprs

    def raw_exprs_to_scores(self, output, length=None):
        scores = output.npvalue()
        length = length or int(math.sqrt(scores.shape[0] + 0.5))
        scores = scores.reshape((length, length))
        return scores

    def get_complete_scores(self, lstm_output):
        length = len(lstm_output)
        output = self.get_complete_raw_exprs(lstm_output)
        scores = self.raw_exprs_to_scores(output, length)
        exprs = self.raw_exprs_to_exprs(output, length)

        return scores, exprs


class LabelEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--label-bilinear-dim", type=int, dest="label_bilinear_dim", default=100)
        group.add_argument("--label-mlp-dims", dest="label_mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[])

    def __init__(self, model, statistics, options):
        super(LabelEvaluation, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]
        self.irels = list(statistics.labels)
        self.rels = {v: idx for idx, v in enumerate(self.irels)}
        self.ldims = options.lstm_dims

        self.relation_bilinear_layer = nn.BiLinear(self, self.ldims * 2,
                                                   options.label_bilinear_dim)
        relation_dense_dims = [options.label_bilinear_dim] + options.label_mlp_dims + \
                              [len(self.irels)]
        if any(i < len(self.irels) for i in [options.label_bilinear_dim] + options.label_mlp_dims):
            logger.warning("Too many labels!")

        self.relation_dense_layer = nn.DenseLayers(self, relation_dense_dims,
                                                   self.activation)

    def restore_components(self, components):
        self.relation_bilinear_layer = components.pop(0)
        self.relation_dense_layer = components.pop(0)
        assert not components

    def get_label_scores(self, lstm_output, edges):
        """
        :type lstm_output: list[dn.Expression]
        :type edges: Edge
        :return:
        """
        rheadfov = [None] * len(lstm_output)
        rmodfov = [None] * len(lstm_output)

        for source, label, target in edges:
            if rheadfov[source] is None:
                rheadfov[source] = self.relation_bilinear_layer.w1.expr() * lstm_output[source]
            if rmodfov[target] is None:
                rmodfov[target] = self.relation_bilinear_layer.w2.expr() * lstm_output[target]

            hidden = self.activation(
                rheadfov[source] + rmodfov[target] +
                self.relation_bilinear_layer.bias.expr())
            output = self.relation_dense_layer(hidden)

            yield output


class EdgeEvaluationNetwork(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--disablelabels", action="store_false", dest="labelsFlag", default=True)
        SentenceEmbeddings.add_parser_arguments(arg_parser)
        EdgeEvaluation.add_parser_arguments(arg_parser)
        LabelEvaluation.add_parser_arguments(arg_parser)

    def __init__(self, model, statistics, options):
        super(EdgeEvaluationNetwork, self).__init__(model)
        self.sent_embedding = SentenceEmbeddings(self, statistics, options)
        self.edge_eval = EdgeEvaluation(self, options)
        self.require_labels = options.labelsFlag

        if self.require_labels:
            self.label_eval = LabelEvaluation(self, statistics, options)
            self.rels = self.label_eval.rels
            self.irels = self.label_eval.irels
        else:
            self.label_eval = None

    def restore_components(self, components):
        self.sent_embedding = components.pop(0)
        self.edge_eval = components.pop(0)
        if self.require_labels:
            self.label_eval = components.pop(0)
        else:
            self.label_eval = None
        assert not components

    def get_vecs(self, node):
        return self.sent_embedding.get_vecs(node)

    def get_lstm_output(self, sentence):
        return self.sent_embedding.get_lstm_output(sentence)

    def get_complete_scores(self, lstm_output):
        return self.edge_eval.get_complete_scores(lstm_output)

    def get_label_scores(self, lstm_output, edges):
        """
        :type lstm_output: list[dn.Expression]
        :type edges: Edge
        :return: 
        """
        return self.label_eval.get_label_scores(lstm_output, edges)
