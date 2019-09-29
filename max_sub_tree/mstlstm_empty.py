import base64
from io import open
import pickle
import random
import time
from collections import namedtuple, Counter, defaultdict
from operator import itemgetter
from typing import List

import dynet as dn
import numpy as np
from six.moves import range

import nn
from conll_reader import SimpleSentence
from conll_reader import SimpleNode
from edge_eval_network import EdgeEvaluationNetwork, EdgeEvaluation
from edge_eval_network3 import EdgeSiblingEvaluation
from logger import logger
from max_sub_tree.decoder import eisner2nd
from max_sub_tree.empty_node import EdgeToEmpty
from max_sub_tree.xeisner import MAX_SENT_SIZE_EMPTY, arcs_with_empty_to_biarcs
from nn import activations
from parser_base import DependencyParserBase
from tree_utils import SentenceNode, Sentence
from utils.eval import eval_empty_tree, eval_empty_node


class SentenceNodeWithEmpty(SentenceNode):
    @classmethod
    def from_simple_node(cls, id_, simple_node):
        """:type simple_node: SimpleNode"""
        ret = cls(id_, simple_node.word,
                  "_", simple_node.postag, simple_node.postag, "_",
                  int(simple_node.head) + 1, simple_node.deprel)
        return ret

    @classmethod
    def tail_node(cls, id):
        return cls(id, '*tail*', '*tail*', 'TAIL-POS', 'TAIL-CPOS', '_', -1, 'rtail')


class SentenceWithEmpty(Sentence): # type: List[SentenceNodeWithEmpty]
    NodeType = SentenceNodeWithEmpty
    EmptyTag = "EMCAT"

    def __init__(self, seq):
        super(SentenceWithEmpty, self).__init__(seq)
        self.empty_nodes = set()

    @classmethod
    def from_simple_sentence(cls, s):
        sentence = cls([cls.NodeType.root_node()])
        sentence_origin_index = list(sentence)
        i = 0
        for node in s:
            if node.postag != cls.EmptyTag:
                i += 1
                node = cls.NodeType.from_simple_node(i, node)
                sentence.append(node)
                sentence_origin_index.append(node)
            else:
                sentence_origin_index.append(None)

        i = 0
        for node in s:
            if node.postag == cls.EmptyTag:
                sentence.empty_nodes.add(EdgeToEmpty(sentence_origin_index[int(node.head) + 1].id,
                                         1, i)) # Only 1 empty node is considered now
            else:
                i += 1

        # sentence.append(cls.NodeType.tail_node(len(sentence)))

        for node_ in sentence[1:]:
            node_.parent_id = sentence_origin_index[node_.parent_id].id

        return sentence

    def to_simple_sentence(self, heads, labels, empty_nodes):
        result = SimpleSentence([])
        id_transform = {0: -1}
        empty_by_position = defaultdict(list)
        for i in empty_nodes:
            empty_by_position[i.position].append(i)
        new_idx = 0
        for idx, node in enumerate(self):
            if idx > 0:
                for i in empty_by_position[idx-1]:
                    result.append(SimpleNode("*", SentenceWithEmpty.EmptyTag, i.head, "xxx"))
                    new_idx += 1
                result.append(SimpleNode(node.form, node.postag, heads[idx], labels[idx]))
                id_transform[idx] = new_idx
                new_idx += 1

        new_result = SimpleSentence([])
        for idx, node in enumerate(result):
            new_result.append(SimpleNode(node.word, node.postag, id_transform[node.head], node.deprel))
        return new_result

    @classmethod
    def from_file(cls, file_name, use_edge=True, root_last=False):
        assert root_last == False
        with open(file_name) as f:
            return [cls.from_simple_sentence(i)
                    for i in SimpleSentence.get_all_sentences(f)]

    @staticmethod
    def evaluate_with_external_program(gold_file, output_file):
        a1 = eval_empty_tree(gold_file, output_file, "Tree")
        try:
            a2 = eval_empty_node(gold_file, output_file, "Tree")
        except:
            a2 = ""
        print(a1)
        print(a2)
        with open(output_file + ".txt", "w") as f:
            f.write(u"{}\n".format(a1))
            f.write(u"{}\n".format(a2))


class StatisticsWithEmpty(namedtuple("_", ["words", "postags", "labels", "emptys"])):
    @classmethod
    def from_sentences(cls, sentences):
        """:type sentences: List[SentenceWithEmpty]"""
        ret = cls(Counter(), Counter(), Counter(), Counter())
        for sentence in sentences:
            ret.words.update(i.norm for i in sentence)
            ret.emptys.update(i[1] for i in sentence.empty_nodes)
            ret.postags.update(i.postag for i in sentence)
            ret.labels.update(i.relation for i in sentence)
        return ret



def emptyeisner(scores, sentence=None):
    from max_sub_tree import xeisner
    empty_count = scores.shape[0] - 1
    N = scores.shape[1] - 1

    scores_n = np.copy(scores)
    if sentence is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if sentence[j].parent_id != i:
                    scores_n[0][i][j] += 1
                if (i, 1, j) not in sentence.empty_nodes:
                    scores_n[1][i][j] += 1

    return xeisner.emptyeisner1st_decode(N, empty_count, scores_n)


def emptyeisner2nd(scores, scores_2, scores_solid, scores_mid, scores_out, sentence=None):
    from max_sub_tree import xeisner
    empty_count = scores.shape[0] - 1
    N = scores.shape[1] - 1

    scores_n = np.copy(scores)
    scores_2_n = np.copy(scores_2)
    scores_solid_n = np.copy(scores_solid)
    scores_mid_n = np.copy(scores_mid)
    scores_out_n = np.copy(scores_out)
    if sentence is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if sentence[j].parent_id != i:
                    scores_n[0, i, j] += 1
                if (i, 1, j) not in sentence.empty_nodes:
                    scores_n[1, i, j] += 1

        biarcs2, biarcs3 = xeisner.arcs_with_empty_to_biarcs([i.parent_id for i in sentence], sentence.empty_nodes)
        for i in range(0, N+1):
            for k in range(1, N+1):
                if (i, k) not in biarcs2:
                    scores_2_n[0, i, k] += 1
                if (i, 1, k) not in biarcs2:
                    scores_2_n[1, i, k] += 1
                for j in range(1, N+1):
                    if (i, j, k) not in biarcs3:
                        scores_solid_n[i, j, k] += 1
                    if (i, (i, 1, j), k) not in biarcs3:
                        scores_mid_n[i, j, k] += 1
                    if (i, j, (i, 1, k)) not in biarcs3:
                        scores_out_n[i, j, k] += 1

    return xeisner.emptyeisner2nd_decode(N, empty_count, scores_n, scores_2_n, scores_solid, scores_mid, scores_out)


def emptyeisnerf(scores, sentence=None):
    from max_sub_tree import xeisner
    empty_count = scores.shape[0] - 1
    N = scores.shape[1] - 1

    scores_n = np.copy(scores)
    if sentence is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if sentence[j].parent_id != i:
                    scores_n[0][i][j] += 1
                if (i, 1, j) not in sentence.empty_nodes:
                    scores_n[1][i][j] += 1

    return xeisner.emptyeisner1stf_decode(N, empty_count, scores_n)


def empty_eisner_greedy(scores, sentence=None):
    from max_sub_tree import xeisner
    N = scores.shape[1] - 1

    scores_n = np.copy(scores)
    if sentence is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if sentence[j].parent_id != i:
                    scores_n[0][i][j] += 1
                if (i, 1, j) not in sentence.empty_nodes:
                    scores_n[1][i][j] += 1

    heads = xeisner.eisner1st_decode(scores_n[0])
    emptys = set()
    for head in range(1, N+1):
        for position in range(0, N+1):
            if scores_n[1][head][position] > 0:
                emptys.add(EdgeToEmpty(head, 1, position))
    return heads, emptys


class TreeWithEmptyTrainer(DependencyParserBase):
    DataType = SentenceWithEmpty

class MaxSubTreeWithEmptyParser(TreeWithEmptyTrainer):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(MaxSubTreeWithEmptyParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--decoder", type=str, dest="decoder", default="eisner2nd", choices=["eisner2nd"])
        group.add_argument("--disable-cost-augment", action="store_false", dest="cost_augment", default=True)
        group.add_argument("--enable-2nd", action="store_true", dest="use_2nd", default=False)

        EdgeEvaluationNetwork.add_parser_arguments(arg_parser)

    def __init__(self, options, train_sentences=None, restore_file=None, statistics=None):
        self.model = dn.Model()

        random.seed(1)
        self.trainer = dn.AdamTrainer(self.model)

        self.activation = activations[options.activation]
        # self.decoder = decoders[options.decoder]

        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.cost_augment
        self.options = options

        if "func" in options:
            del options.func

        if restore_file:
            self.container, = dn.load(restore_file, self.model)
            networks = list(self.container.components)
            self.network = networks.pop(0)
            self.statistics = statistics
            self.has_emptys = len(statistics.emptys) > 0
            if self.has_emptys:
                self.network_for_emptys = networks.pop(0)
            if self.options.use_2nd:
                self.network3 = networks.pop(0)
                if self.has_emptys:
                    self.network3_for_emptys_mid = networks.pop(0)
                    self.network3_for_emptys_out = networks.pop(0)
            assert not networks
        else:
            self.container = nn.Container(self.model)
            self.statistics = statistics = StatisticsWithEmpty.from_sentences(train_sentences)
            self.has_emptys = len(statistics.emptys) > 0
            self.network = EdgeEvaluationNetwork(self.container, statistics, options)
            if self.has_emptys:
                self.network_for_emptys = EdgeEvaluation(self.container, options)
            if options.use_2nd:
                self.network3 = EdgeSiblingEvaluation(self.container, options)
                if self.has_emptys:
                    self.network3_for_emptys_mid = EdgeSiblingEvaluation(self.container, options)
                    self.network3_for_emptys_out = EdgeSiblingEvaluation(self.container, options)

    def predict(self, sentence):
        """ :type sentence: SentenceWithEmpty"""
        for iSentence, sentence in enumerate(sentence):
            if len(sentence) >= MAX_SENT_SIZE_EMPTY - 1:
                logger.info("sent too long...")
                heads = [0 for _ in range(len(sentence))]
                labels = [None for _ in range(len(sentence))]
                yield sentence.to_simple_sentence(heads, labels, set())
                continue

            lstm_output = self.network.get_lstm_output(sentence)
            scores, exprs = self.network.get_complete_scores(lstm_output)
            labels = [None for _ in range(len(sentence))]
            if self.has_emptys:
                scores_ec, exprs_ec = self.network_for_emptys.get_complete_scores(lstm_output)
            else:
                scores_ec = -np.ones((len(lstm_output), len(lstm_output)), dtype=np.float64)

            scores_all = np.stack([scores, scores_ec])
            if self.options.use_2nd:
                exprs2nd2, scores2nd2, exprs2nd3, scores2nd3 = self.network3.get_complete_scores(lstm_output)
                if self.has_emptys:
                    _, _, exprs_mid, scores_mid = self.network3_for_emptys_mid.get_complete_scores(lstm_output, False)
                    exprs2nd2_ec, scores2nd2_ec, exprs_out, scores_out = self.network3_for_emptys_out.get_complete_scores(
                        lstm_output)
                    scores_all_ec = np.stack([scores2nd2, scores2nd2_ec])
                    heads, emptys = emptyeisner2nd(scores_all, scores_all_ec, scores2nd3, scores_mid, scores_out)
                else:
                    heads = eisner2nd(scores, scores2nd2, scores2nd3)
                    emptys = set()
            else:
                heads, emptys = empty_eisner_greedy(scores_all)

            if self.labelsFlag:
                edges = [(head, "_", modifier) for modifier, head in enumerate(heads[1:], 1)]
                for edge, scores_expr in \
                        zip(edges, self.network.get_label_scores(lstm_output, edges)):
                    head, _, modifier = edge
                    labels_scores = scores_expr.value()
                    labels[modifier] = \
                        self.network.irels[max(enumerate(labels_scores), key=itemgetter(1))[0]]

            dn.renew_cg()

            result = sentence.to_simple_sentence(heads, labels, emptys)
            if self.options.output_scores:
                # extract full edge scores
                if self.has_emptys:
                    scores_list = [scores, scores_ec, scores2nd2, scores2nd3,
                                   scores2nd2_ec, scores_mid, scores_out]
                else:
                    scores_list = [scores, None, scores2nd2, scores2nd3,
                                   None, None, None]
                result.comment = [base64.b64encode(pickle.dumps(scores_list)).decode()]
            yield result

    # noinspection PyUnboundLocalVariable
    def train_gen(self, sentences, update=True):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        errs = []
        lerrs = []

        empty_correct_count = 0
        empty_gold_total = 0.00000001
        empty_pred_total = 0.00000001

        for sent_idx, sentence in enumerate(sentences):
            if sent_idx % 100 == 0 and sent_idx != 0:
                logger.info("Processing sentence number: %d, Loss: %.2f,"
                            "Empty-R: %.2f, Empty-P: %.2f, Errors: %.2f, Time: %.2f",
                            sent_idx, eloss / etotal, empty_correct_count / empty_gold_total,
                            empty_correct_count / empty_pred_total,
                            (float(eerrors)) / etotal, time.time() - start)
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                empty_gold_total = 0.00000001
                empty_pred_total = 0.00000001
                empty_correct_count = 0

            if len(sentence) >= MAX_SENT_SIZE_EMPTY - 1:
                logger.info("sent too long...")
                continue

            empty_gold_total += len(sentence.empty_nodes)
            lstm_output = self.network.get_lstm_output(sentence)
            scores, exprs = self.network.get_complete_scores(lstm_output)
            if self.has_emptys:
                scores_ec, exprs_ec = self.network_for_emptys.get_complete_scores(lstm_output)
            else:
                scores_ec = -np.ones((len(lstm_output), len(lstm_output)), dtype=np.float64)

            gold = [entry.parent_id for entry in sentence]
            scores_all = np.stack([scores, scores_ec])

            if self.options.use_2nd:
                exprs2nd2, scores2nd2, exprs2nd3, scores2nd3 = self.network3.get_complete_scores(lstm_output)
                if self.has_emptys:
                    _, _, exprs_mid, scores_mid = self.network3_for_emptys_mid.get_complete_scores(lstm_output, False)
                    exprs2nd2_ec, scores2nd2_ec, exprs_out, scores_out = \
                        self.network3_for_emptys_out.get_complete_scores(lstm_output)
                    scores_all_ec = np.stack([scores2nd2, scores2nd2_ec])
                    heads, emptys = emptyeisner2nd(scores_all, scores_all_ec, scores2nd3, scores_mid, scores_out,
                                                   sentence if self.costaugFlag else None)
                else:
                    heads = eisner2nd(scores, scores2nd2, scores2nd3, gold if self.costaugFlag else None)
                    emptys = set()
            else:
                heads, emptys = empty_eisner_greedy(scores_all, sentence if self.costaugFlag else None)
            empty_pred_total += len(emptys)

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

            if self.has_emptys:
                empty_incorrect = sentence.empty_nodes.symmetric_difference(emptys)
                empty_correct = sentence.empty_nodes.intersection(emptys)
                empty_correct_count += len(empty_correct)
                for i in empty_incorrect:
                    if i not in sentence.empty_nodes:
                        errs.append(exprs_ec[i.head][i.position])
                    else:
                        errs.append(-exprs_ec[i.head][i.position])

            if self.options.use_2nd:
                # noinspection PyUnboundLocalVariable
                biarcs2, biarcs3 = arcs_with_empty_to_biarcs(heads, emptys)
                biarcs2_gold, biarcs3_gold = arcs_with_empty_to_biarcs(gold, sentence.empty_nodes)
                for s, t in biarcs2.symmetric_difference(biarcs2_gold):
                    sign = 1 if (s, t) in biarcs2 else -1
                    if isinstance(t, int):
                        errs.append(exprs2nd2[s, t] * sign)
                    elif isinstance(t, EdgeToEmpty):
                        errs.append(exprs2nd2_ec[s, t.position] * sign)

                for s, m, t in biarcs3.symmetric_difference(biarcs3_gold):
                    sign = 1 if (s, m, t) in biarcs3 else -1
                    if isinstance(m, int) and isinstance(t, int):
                        errs.append(exprs2nd3[s, m, t] * sign)
                    elif isinstance(m, EdgeToEmpty) and isinstance(t, int):
                        errs.append(exprs_mid[s, m.position, t] * sign)
                    elif isinstance(m, int) and isinstance(t, EdgeToEmpty):
                        errs.append(exprs_out[s, m, t.position] * sign)
                    else:
                        raise TypeError

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
            pickle.dump((self.options, self.statistics), f)
        # noinspection PyArgumentList
        dn.save(prefix, [self.container])

    @classmethod
    def load(cls, prefix, new_options=None):
        """
        :param prefix: model file name prefix
        :type prefix: str
        :rtype: MaxSubTreeWithEmptyParser
        """
        with open(prefix + ".options", "rb") as f:
            options, statistics = pickle.load(f)
        options.__dict__.update(new_options.__dict__)
        ret = cls(options, None, prefix, statistics)
        return ret
