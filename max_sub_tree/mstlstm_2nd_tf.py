from __future__ import division

import gzip
import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl

import itertools
from conll_reader import CoNLLUSentence, CoNLLUNode
from common_utils import Timer
from vocab_utils import Dictionary, Statistics
from edge_eval_network_tf import SentenceEmbeddingLayer, InteractionBilinearLayer, EdgeEvaluationLayer, \
    EdgeDecoderLayer, LossCalculationLayer, LabelLossCalculationLayer
from max_sub_tree.decoder import eisner
from parser_base import TreeParserBase
from logger import logger

from pathos.multiprocessing import ProcessingPool

pool = ProcessingPool(6)


def tf_interface(func):
    func_w = lambda args: func(*args)

    def wrapped(scores, lengths, gold_edges=()):
        args = []
        for sent_id, (scores_0, length) in enumerate(
                itertools.zip_longest(scores, lengths)):
            if gold_edges != ():
                heads = np.zeros((length, ), dtype=np.int32)
                for sent_id_2, head, dep in gold_edges:
                    if sent_id == sent_id_2:
                        heads[dep] = head
                args.append((scores_0[:length, :length], heads))
            else:
                args.append((scores_0[:length, :length],))

        a_results = pool.amap(func_w, args)
        results = a_results.get()
        edges = np.array([[sent_id, head, dep]
                          for sent_id, heads in enumerate(results)
                          for dep, head in enumerate(heads[1:], 1)
                          if head != -1],
                         dtype=np.int32)
        return edges

    return wrapped


eisner_tf = tf_interface(eisner)


def get_external_embedding(embedding_filename, encoding="utf-8"):
    ext_dict = Dictionary(initial=("___PAD___", "___UNKNOWN___"))

    def read_embedding(fp):
        for line in fp:
            fields = line.decode(encoding).strip().split(' ')
            if len(fields) <= 2:
                continue
            token = fields[0]
            vector = [float(i) for i in fields[1:]]
            yield token, vector

    if embedding_filename.endswith(".gz"):
        external_embedding_fp = gzip.open(embedding_filename, 'rb')
    else:
        external_embedding_fp = open(embedding_filename, 'rb')

    words, embeddings = zip(*list(read_embedding(external_embedding_fp)))
    external_embedding_fp.close()

    dims = len(embeddings[0])
    embeddings_np = np.array(([0] * dims, [0] * dims) + embeddings, dtype=np.float32)

    ext_dict.update(words)
    return embeddings_np, ext_dict


class MaxSubTreeParser(TreeParserBase):
    MAX_SENT_LEN = 130
    BUCKET_RANGE = 25
    BATCH_SIZE = 16
    PRINT_PER = 100

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(MaxSubTreeParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--decoder", type=str, dest="decoder", default="eisner2nd", choices=["eisner"])
        group.add_argument("--cost-augment", action="store_true", dest="cost_augment", default=True)

        SentenceEmbeddingLayer.add_parser_arguments(arg_parser)
        InteractionBilinearLayer.add_parser_arguments(arg_parser)
        EdgeEvaluationLayer.add_parser_arguments(arg_parser)

    def __init__(self, options, train_sentences=None, restore_file=None):
        self.statistics = Statistics.from_sentences(train_sentences)

        if options.external is not None:
            ext_embedding, self.ext_dict = get_external_embedding(options.external)
        else:
            ext_embedding, self.ext_dict = None, None

        self.cgs = {}  # computational graphs
        x_lengths = tf.placeholder(tf.int32, shape=(None,), name="x_length")

        reuse = False
        for sent_len in range(self.BUCKET_RANGE, self.MAX_SENT_LEN + 1, self.BUCKET_RANGE):
            with tf.variable_scope("model", reuse=reuse):
                x_words = tf.placeholder(tf.int32, shape=(None, sent_len), name="x_words")
                x_pos = tf.placeholder(tf.int32, shape=(None, sent_len), name="x_pos")
                y_edge = tf.placeholder(tf.int32, shape=(None, 3), name="y_edge")
                y_label = tf.placeholder(tf.int32, shape=(None, 4), name="y_label")
                if options.external is not None:
                    x_ext = tf.placeholder(tf.int32, shape=(None, sent_len), name="x_ext")
                    network = SentenceEmbeddingLayer(sent_len, x_words, x_pos, x_lengths,
                                                     self.statistics, options, ext_embedding, x_ext)
                else:
                    x_ext = None
                    network = SentenceEmbeddingLayer(sent_len, x_words, x_pos, x_lengths, self.statistics, options)
                edge_bilinear = InteractionBilinearLayer(network, options, name="edge_bilinear")
                edge_scores = EdgeEvaluationLayer(edge_bilinear, options, 1, False, name="edge_scores")
                label_bilinear = InteractionBilinearLayer(network, options, name="label_bilinear")
                label_scores = EdgeEvaluationLayer(label_bilinear, options,
                                                   len(self.statistics.labels), True, name="label_scores")
                decoded = EdgeDecoderLayer(edge_scores, x_lengths, y_edge, eisner_tf, name="decoded")
                decoded_test = EdgeDecoderLayer(edge_scores, x_lengths, None, eisner_tf, name="decoded_test")
                loss_edge = LossCalculationLayer(edge_scores, decoded, x_lengths, y_edge, name="loss_edge")
                loss, uas, loss_shift = loss_edge.outputs

                loss_label_layer = LabelLossCalculationLayer(label_scores, y_edge, y_label)
                label_loss, labels = loss_label_layer.outputs

                train_op = tf.train.AdamOptimizer(options.learning_rate).minimize(loss + label_loss)
                self.cgs[sent_len] = (x_words, x_pos, x_lengths, y_edge, loss, train_op, uas,
                                      loss_shift, decoded_test, x_ext, y_label, label_loss, labels, label_scores.outputs)
                tl.layers.set_name_reuse(True)
                reuse = True
        tl.layers.set_name_reuse(False)
        self.session = tf.Session()
        tl.layers.initialize_global_variables(self.session)

    def train(self, sentences):
        sentences_ = [i for i in sentences if len(i) <= self.MAX_SENT_LEN]
        if len(sentences_) != len(sentences):
            logger.warning("Ignore {} long sentences(> {}).".format(
                len(sentences) - len(sentences_), self.MAX_SENT_LEN))
        timer = Timer()
        for sent_idx in range(0, len(sentences_), self.BATCH_SIZE):
            sent_batch = sentences_[sent_idx:sent_idx + self.BATCH_SIZE]
            lengths = np.array([len(i) for i in sent_batch]) # length including root
            max_length = np.max(lengths)
            bucket = int(math.ceil(max_length / self.BUCKET_RANGE) * self.BUCKET_RANGE)
            assert bucket >= max_length

            edges_indices = np.array([[sent_id, head, dep]
                              for sent_id, sent in enumerate(sent_batch)
                              for dep, head in
                              enumerate((i.parent_id for i in sent[1:]), 1)])

            labels_indices = np.array([[sent_id, head, dep, self.statistics.labels.word_to_int[relation]]
                                          for sent_id, sent in enumerate(sent_batch)
                                          for dep, (head, relation) in
                                          enumerate(((i.parent_id, i.relation) for i in sent[1:]), 1)])

            words = self.statistics.words.lookup(sent_batch, bucket, "norm")
            postags = self.statistics.postags.lookup(sent_batch, bucket, "postag")
            x_words, x_pos, x_lengths, y_edge, loss_edge, train_op, uas, loss_shift, decoded_test, \
            x_ext, y_label, label_loss, labels, label_output = self.cgs[bucket]
            feed_dict = {x_words: words, x_pos: postags, x_lengths: lengths,
                         y_edge: edges_indices, y_label: labels_indices}
            if self.ext_dict:
                feed_dict[x_ext] = self.ext_dict.lookup(sent_batch, bucket, ("form", "norm"))
            loss, n_uas, _ = self.session.run([loss_edge, uas, train_op], feed_dict)

            if sent_idx % MaxSubTreeParser.PRINT_PER == 0:
                print("Sentence {}-{}, Loss: {:.2f}, UAS:{:.2f}, Time: {}".format(
                    sent_idx, sent_idx + self.BATCH_SIZE - 1,
                    loss, n_uas, timer.tick()))

    def predict(self, sentences):
        timer = Timer()
        for sent_idx in range(0, len(sentences), self.BATCH_SIZE):
            sent_batch = sentences[sent_idx:sent_idx + self.BATCH_SIZE]
            lengths = np.array([len(i) for i in sent_batch])
            max_length = np.max(lengths)
            bucket = int(math.ceil(max_length / self.BUCKET_RANGE) * self.BUCKET_RANGE)
            assert bucket >= max_length
            words = self.statistics.words.lookup(sent_batch, bucket, "norm")
            postags = self.statistics.postags.lookup(sent_batch, bucket, "postag")
            x_words, x_pos, x_lengths, y_edge, loss_edge, train_op, uas, loss_shift, decoded_test,\
            x_ext, y_label, label_loss, labels, label_output = self.cgs[bucket]
            feed_dict = {x_words: words, x_pos: postags, x_lengths: lengths}
            if self.ext_dict:
                feed_dict[x_ext] = self.ext_dict.lookup(sent_batch, bucket, ("form", "norm"))
            decoded_, labels_scores = self.session.run([decoded_test.outputs, label_output], feed_dict)
            labels_ = np.argmax(labels_scores, axis=3)
            heads_dict = {(sent_idx2, dep): head for sent_idx2, head, dep in decoded_}

            for sent_idx2, sentence in enumerate(sent_batch):
                def convert_node(node):
                    head = heads_dict[sent_idx2, node.id]
                    return CoNLLUNode(node.id, node.form, node.lemma, node.cpos,
                                      node.pos, node.feats,
                                      head, self.statistics.labels.int_to_word[labels_[sent_idx2, head, node.id]],
                                      "_", "_")

                yield CoNLLUSentence(convert_node(node) for node in sentence if node.id > 0)

    def save(self, prefix):
        pass

    @classmethod
    def load(cls, prefix):
        pass
