from __future__ import unicode_literals
from io import open

import gzip

import numpy as np
import six
import math
import tensorflow as tf
import tensorlayer as tl

import tl_mod
from logger import logger


class ExternalEmbeddingLayer(tl.layers.Layer):
    def __init__(self, external, x_ext, name="ext_embedding"):
        super(ExternalEmbeddingLayer, self).__init__(name=name)
        embeddings = tf.get_variable(name='ext_embeddings',
                     initializer=external)
        embed = tf.nn.embedding_lookup(embeddings, x_ext)
        self.outputs = embed

        print("  [TL] ExternalEmbeddingInputlayer %s: " % embeddings.shape)

        self.all_layers = [self.outputs]
        self.all_params = [embed]
        self.all_drop = {}


class SentenceEmbeddingLayer(tl.layers.Layer):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--activation", type=str, dest="activation", default="tanh")
        group.add_argument("--word-embedding", type=int, dest="w_dims", default=100)
        group.add_argument("--pos-embedding", type=int, dest="p_dims", default=25)
        group.add_argument("--extrn", dest="external", help="External embeddings", metavar="FILE")
        group.add_argument("--rnn-layer-count", type=int, dest="rnn_count", default=2)
        group.add_argument("--rnn-units", type=int, dest="rnn_units", default=125)

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):
        super(SentenceEmbeddingLayer, self).__init__()
        raise NotImplementedError

    def __new__(cls, n, x_words, x_pos, x_lengths, statistics, options, external=None, x_ext=None):
        w_embedding = tl.layers.EmbeddingInputlayer(x_words, len(statistics.words),
                                                    options.w_dims, name="w_embedding")
        p_embedding = tl.layers.EmbeddingInputlayer(x_pos, len(statistics.postags),
                                                    options.p_dims, name="p_embedding")
        if external is not None:
            e_embedding = ExternalEmbeddingLayer(external, x_ext)
            all_embeddings = tl.layers.ConcatLayer([w_embedding, p_embedding, e_embedding],
                                                   concat_dim=2,
                                                   name="all_embeddings")
        else:
            all_embeddings = tl.layers.ConcatLayer([w_embedding, p_embedding],
                                                   concat_dim=2,
                                                   name="all_embeddings")

        network = tl_mod.BiDynamicRNNLayerM(all_embeddings, tf.contrib.rnn.BasicLSTMCell,
                                            n_hidden=options.rnn_units,
                                            n_layer=options.rnn_count,
                                            sequence_length=x_lengths,
                                            name="birnn")

        return network


class InteractionBilinearLayer(tl.layers.Layer):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--bilinear_dim", type=int, dest="bilinear_dim", default=100)

    def __init__(self, layer, options, name="bilinear"):
        super(InteractionBilinearLayer, self).__init__(name=name)

        n = layer.outputs.shape[1].value
        rnn_units = layer.outputs.shape[2].value
        self.inputs = layer.outputs
        inputs_1d = tf.reshape(self.inputs, (-1, rnn_units))

        w1 = tf.get_variable(name=self.name + "/w1", shape=(rnn_units, options.bilinear_dim))
        w2 = tf.get_variable(name=self.name + "/w2", shape=(rnn_units, options.bilinear_dim))
        b = tf.get_variable(name=self.name + "/b", shape=(options.bilinear_dim,))

        inputs_w1 = tf.reshape(tf.matmul(inputs_1d, w1), (-1, n, options.bilinear_dim), name="input_w1")
        inputs_w2 = tf.reshape(tf.matmul(inputs_1d, w2), (-1, n, options.bilinear_dim), name="input_w2")

        mods_part = tf.tile(inputs_w2, (1, n, 1), name="tile_w2")  # w2 * [1 2 3 1 2 3 1 2 3]

        m0 = tf.tile(inputs_w1, (1, n, 1), name="tile_w1")  # w1 * [1 2 3 1 2 3 1 2 3]
        m1 = tf.reshape(m0, (-1, n, n, options.bilinear_dim), name="m1")  # w1 * [[1 2 3][1 2 3][1 2 3]]
        m2 = tf.transpose(m1, perm=[0, 2, 1, 3], name="m2")  # w1 * [[1 1 1][2 2 2][3 3 3]]
        heads_part = tf.reshape(m2, (-1, n * n, options.bilinear_dim), name="heads_part")  # w1 * [1 1 1 2 2 2 3 3 3]

        self.outputs = getattr(tf.nn, options.activation)(heads_part + mods_part + b)

        # mods_part, heads_part = tf.meshgrid(inputs_w1, inputs_w2)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])
        self.all_params.extend([w1, w2, b])


class EdgeEvaluationLayer(tl.layers.Layer):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--mlp-dims", dest="mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[])

    def __init__(self, layer, options, output_count, last_bias, name="edge_eval"):
        super(EdgeEvaluationLayer, self).__init__(name=name)
        self.inputs = layer.outputs
        n = int(math.sqrt(self.inputs.shape[1].value + 0.1))
        bilinear_dim = self.inputs.shape[2].value

        layer = tl.layers.ReshapeLayer(layer, (-1, bilinear_dim), name="{}-reshape-1d".format(name))
        for idx, num_units in enumerate(options.mlp_dims):
            layer = tl.layers.DenseLayer(layer,
                                         n_units=num_units,
                                         act=getattr(tf.nn, options.activation),
                                         name="{}-dense-{}".format(name, idx))

        layer = tl.layers.DenseLayer(layer,
                                     n_units=output_count,
                                     b_init=tf.constant_initializer(value=0.0) if last_bias else None,
                                     name="{}-dense-{}".format(name, len(options.mlp_dims)))

        if output_count == 1:
            layer = tl.layers.ReshapeLayer(layer, (-1, n, n), name="{}-reshape-back".format(name))
        else:
            layer = tl.layers.ReshapeLayer(layer, (-1, n, n, output_count), name="{}-reshape-back".format(name))
        self.outputs = layer.outputs

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])


class EdgeDecoderLayer(tl.layers.Layer):
    def __init__(self, layer, x_length, y_, decoder, name="edge_decode"):
        super(EdgeDecoderLayer, self).__init__(name=name)
        self.inputs = layer.outputs
        decoded = tf.py_func(decoder,
                             [self.inputs, x_length, y_] if y_ is not None else [self.inputs, x_length],
                             tf.int32, name="decoder_py_function")
        self.outputs = decoded

        self.outputs.set_shape((None, 3))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])


class LossCalculationLayer(tl.layers.Layer):
    @staticmethod
    def count_equal(a, b):
        edges_a = set(tuple(i) for i in a)
        edges_b = set(tuple(i) for i in b)
        correct = len(edges_a.intersection(edges_b))
        total = len(edges_a) + 0.0001
        return np.float32(correct / total)

    def __init__(self, scores_layer, decoder_layer, x_length, y_, name="loss_calc"):
        super(LossCalculationLayer, self).__init__(name=name)
        predicted_edges = decoder_layer.outputs
        gold_edges = y_
        uas = tf.py_func(LossCalculationLayer.count_equal, [predicted_edges, gold_edges], tf.float32)
        loss = tf.reduce_sum(tf.gather_nd(scores_layer.outputs, predicted_edges)) - \
            tf.reduce_sum(tf.gather_nd(scores_layer.outputs, gold_edges))
        loss_shift = 0
        self.outputs = loss, uas, loss_shift


class LabelLossCalculationLayer(tl.layers.Layer):
    @staticmethod
    def count_equal(pred_edge_labels, gold_label_indices):
        total = len(gold_label_indices) + 0.0001
        correct = 0
        for batch_idx, head, dep, label in pred_edge_labels:
            if pred_edge_labels[batch_idx, head, dep] == label:
                correct += 1
        return np.float32(correct / total)

    @staticmethod
    def loss_augment_decode(full_edge_scores, edge_indices, gold_label_indices):
        for index in gold_label_indices:
            full_edge_scores[tuple(index)] -= 1
        pred_edge_labels = np.argmax(full_edge_scores, axis=3)
        ret =  np.array([tuple(index) + (pred_edge_labels[tuple(index)], ) for index in edge_indices], dtype=np.int32)
        return ret

    def __init__(self, label_scores_layer, edge_indices, gold_label_indices, name="label_loss_calc"):
        super(LabelLossCalculationLayer, self).__init__(name)
        full_edge_scores = self.inputs = label_scores_layer.outputs
        pred_label_indices = tf.py_func(LabelLossCalculationLayer.loss_augment_decode,
                                        [full_edge_scores, edge_indices, gold_label_indices],
                                        tf.int32)
        pred_label_score = tf.reduce_sum(tf.gather_nd(full_edge_scores, pred_label_indices))
        gold_label_score = tf.reduce_sum(tf.gather_nd(full_edge_scores, gold_label_indices))

        loss = pred_label_score - gold_label_score
        self.outputs = loss, pred_label_indices
