import ctypes
import pickle
import random

import dynet as dn
import time
from operator import itemgetter
from six.moves import range

import nn
from conll_reader import CoNLLUSentence, CoNLLUNode
from edge_eval_network import EdgeEvaluationNetwork
from vocab_utils import Statistics
from edge_eval_network3 import EdgeSiblingEvaluation
from logger import logger
from max_sub_tree.decoder import decoders
from max_sub_tree.xeisner import arcs_to_biarcs, MAX_SENT_SIZE_NOEMPTY
from nn import activations
from parser_base import TreeParserBase


class MaxSubTreeParser(TreeParserBase):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(MaxSubTreeParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--decoder", type=str, dest="decoder", default="eisner2nd", choices=["eisner2nd"])
        group.add_argument("--cost-augment", action="store_true", dest="cost_augment", default=True)

        EdgeEvaluationNetwork.add_parser_arguments(arg_parser)

    def __init__(self, options, train_sentences=None, restore_file=None):
        self.model = dn.Model()
        random.seed(1)
        self.trainer = dn.AdamTrainer(self.model)

        self.activation = activations[options.activation]
        self.decoder = decoders[options.decoder]

        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.cost_augment
        self.options = options

        if "func" in options:
            del options.func

        if restore_file:
            self.container, = dn.load(restore_file, self.model)
            self.network, self.network3 = self.container.components
        else:
            self.container = nn.Container(self.model)
            statistics = Statistics.from_sentences(train_sentences)
            self.network = EdgeEvaluationNetwork(self.container, statistics, options)
            self.network3 = EdgeSiblingEvaluation(self.container, options)

    def predict(self, sentence):
        def convert_node(node, heads, labels):
            return CoNLLUNode(node.id, node.form, node.lemma, node.cpos,
                              node.pos, node.feats,
                              heads[node.id], labels[node.id],
                              "_", "_")

        for iSentence, sentence in enumerate(sentence):
            if len(sentence) >= MAX_SENT_SIZE_NOEMPTY - 1:
                logger.info("sent too long...")
                heads = [0 for _ in range(len(sentence))]
                heads[0] = -1
                labels = [None for _ in range(len(sentence))]
                yield CoNLLUSentence(convert_node(node, heads, labels)
                                     for node in sentence if node.id > 0)
                continue
            lstm_output = self.network.get_lstm_output(sentence)
            scores, exprs = self.network.get_complete_scores(lstm_output)
            exprs2nd2, scores2nd2, exprs2nd3, scores2nd3 = self.network3.get_complete_scores(lstm_output)
            heads = self.decoder(scores, scores2nd2, scores2nd3)
            labels = [None for _ in range(len(sentence))]

            if self.labelsFlag:
                edges = [(head, "_", modifier) for modifier, head in enumerate(heads[1:], 1)]
                for edge, scores_expr in \
                        zip(edges, self.network.get_label_scores(lstm_output, edges)):
                    head, _, modifier = edge
                    scores = scores_expr.value()
                    labels[modifier] = \
                        self.network.irels[max(enumerate(scores), key=itemgetter(1))[0]]

            dn.renew_cg()

            yield CoNLLUSentence(convert_node(node, heads, labels)
                                 for node in sentence if node.id > 0)

    def train_gen(self, sentences, update=True):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        errs = []
        lerrs = []

        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) >= MAX_SENT_SIZE_NOEMPTY - 1:
                logger.info("sent too long...")
                continue

            if sent_idx % 100 == 0 and sent_idx != 0:
                logger.info('Processing sentence number: %d, Loss: %.2f, Errors: %.2f, Time: %.2f',
                            sent_idx, eloss / etotal, (float(eerrors)) / etotal, time.time() - start)
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0

            lstm_output = self.network.get_lstm_output(sentence)
            scores, exprs = self.network.get_complete_scores(lstm_output)
            exprs2nd2, scores2nd2, exprs2nd3, scores2nd3 = self.network3.get_complete_scores(lstm_output)

            gold = [entry.parent_id for entry in sentence]
            heads = self.decoder(scores, scores2nd2, scores2nd3, gold if self.costaugFlag else None)

            if self.labelsFlag:
                edges = [(head, "_", modifier) for modifier, head in enumerate(gold[1:], 1)]
                for edge, r_scores_expr in \
                        zip(edges, self.network.get_label_scores(lstm_output, edges)):
                    head, _, modifier = edge
                    r_scores = r_scores_expr.value()
                    gold_label_index = self.network.rels[sentence[modifier].relation]
                    wrong_label_index = max(((l, scr) for l, scr in enumerate(r_scores)
                                             if l != gold_label_index), key=itemgetter(1))[0]
                    if r_scores[gold_label_index] < r_scores[wrong_label_index] + 1:
                        lerrs.append(
                            r_scores_expr[wrong_label_index] -
                            r_scores_expr[gold_label_index])

            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            eerrors += e
            if e > 0:
                loss = [(exprs[h][i] - exprs[g][i]) for i, (h, g) in enumerate(zip(heads, gold)) if
                        h != g]  # * (1.0/float(e))
                eloss += e
                mloss += e
                errs.extend(loss)

            heads2nd2, heads2nd3 = arcs_to_biarcs(heads)
            gold2nd2, gold2nd3 = arcs_to_biarcs(gold)
            for i, k in heads2nd2:
                errs.append(exprs2nd2[i, k])
            for i, k in gold2nd2:
                errs.append(-exprs2nd2[i, k])
            for i, j, k in heads2nd3:
                errs.append(exprs2nd3[i, j, k])
            for i, j, k in gold2nd3:
                errs.append(-exprs2nd3[i, j, k])

            etotal += len(sentence)

            if errs or lerrs:
                loss = dn.esum(errs + lerrs)  # * (1.0/(float(len(errs))))
            else:
                loss = dn.scalarInput(0.0)
            loss_value = loss.scalar_value()
            errs = []
            lerrs = []
            if loss_value != 0.0:
                if update:
                    loss.backward()
                    self.trainer.update()
                    dn.renew_cg()
            yield (loss if not update else None)

    def train(self, sentences):
        for _ in self.train_gen(sentences):
            pass

    def save(self, prefix):
        with open(prefix + ".options", "wb") as f:
            pickle.dump(self.options, f)
        # noinspection PyArgumentList
        dn.save(prefix, [self.network])

    @classmethod
    def load(cls, prefix, new_optons=None):
        """
        :param prefix: model file name prefix
        :type prefix: str
        :rtype: MaxSubGraphParser
        """
        with open(prefix + ".options") as f:
            options = pickle.load(f)
        if new_optons is not None:
            options.__dict__.update(new_optons)
        ret = cls(options, None, prefix)
        return ret
