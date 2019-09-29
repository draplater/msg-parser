from __future__  import division

from collections import namedtuple
from itertools import chain, islice

import dynet as dn
from six.moves import range

from logger import logger
from conll_reader import CoNLLUSentence, CoNLLUNode

History = namedtuple("History", ["score", "correct"])
StateCandidate = namedtuple("StateCandidate", ["state_idx", "joint_idx", "score", "local_score"])


class ArcHybridState(object):
    def __init__(self, sentence, options, copy=False):
        self.sentence = sentence
        self.options = options

        if not copy:
            self.pred_heads = [None for _ in range(1+len(self.sentence))]
            self.pred_labels = [None for _ in range(1+len(self.sentence))]
            self.left_child_ids = [_ for _ in range(1 + len(self.sentence))]
            self.leftmost_child_ids = [_ for _ in range(1 + len(self.sentence))]
            self.right_child_ids = [_ for _ in range(1 + len(self.sentence))]
            self.rightmost_child_ids = [_ for _ in range(1 + len(self.sentence))]
            self.next_word = 0
            self.score = 0
            self.history = []
            self.stack = []
            self.lstm_map = None
            self.is_correct = None

    def set_lstm_map(self, lstm_map):
        self.lstm_map = lstm_map

    def copy_for_foresee(self):
        other = self.__class__(self.sentence, self.options)
        other.stack = list(self.stack)
        other.next_word = self.next_word
        return other

    def copy(self):
        other = self.__class__(self.sentence, self.options, True)
        other.stack = list(self.stack)
        other.history = list(self.history)
        other.score = self.score
        other.next_word = self.next_word
        other.lstm_map = self.lstm_map
        other.is_correct = self.is_correct
        other.pred_heads = list(self.pred_heads)
        other.pred_labels = list(self.pred_labels)
        other.left_child_ids = list(self.left_child_ids)
        other.leftmost_child_ids = list(self.leftmost_child_ids)
        other.right_child_ids = list(self.right_child_ids)
        other.rightmost_child_ids = list(self.rightmost_child_ids)
        return other

    @property
    def is_stack_empty(self):
        return len(self.stack) == 0

    @property
    def is_buffer_empty(self):
        return self.next_word == len(self.sentence)

    @property
    def is_buffer_root(self):
        return self.next_word == len(self.sentence) - 1

    def is_finished(self):
        return len(self.stack) == 0 and self.is_buffer_root

    def get_stack_top_k(self, k, empty_pad):
        return [self.total_features(self.stack[-i-1].id)
                if len(self.stack) > i else [empty_pad] for i in range(k)]

    def total_features(self, index):
        ret = []
        head_lstm = self.lstm_map[index]
        if self.options.headFlag:
            ret.append(head_lstm)
        if self.options.rlFlag:
            left_id = self.left_child_ids[index]
            right_id = self.right_child_ids[index]
            ret.append(self.lstm_map[left_id])
            ret.append(self.lstm_map[right_id])
        if self.options.rlMostFlag:
            leftmost_id = self.leftmost_child_ids[index]
            rightmost_id = self.rightmost_child_ids[index]
            ret.append(self.lstm_map[leftmost_id])
            ret.append(self.lstm_map[rightmost_id])
        return ret

    @property
    def next_word_root(self):
        return self.sentence[self.next_word]


    def get_buffer_top(self, empty_pad):
        if self.is_buffer_empty:
            return [empty_pad]
        return [self.total_features(self.next_word_root.id)]

    def get_input_tensor(self, k, empty_pad):
        topStack = self.get_stack_top_k(k, empty_pad)
        topBuffer = self.get_buffer_top(empty_pad)
        return dn.concatenate(list(chain(*(topStack + topBuffer))))

    @property
    def alpha(self):
        return islice(self.stack, 0, len(self.stack) - 2) if len(self.stack) > 2 else iter(())

    @property
    def s1(self):
        return islice(self.stack, len(self.stack) - 2, len(self.stack) - 1) if len(self.stack) > 1 else iter(())

    @property
    def s0(self):
        return islice(self.stack, len(self.stack) - 1, None) if len(self.stack) > 0 else iter(())

    @property
    def b(self):
        return islice(self.sentence, self.next_word, self.next_word + 1) if not self.is_buffer_empty else iter(())

    @property
    def beta(self):
        return islice(self.sentence, self.next_word + 1, None) \
            if len(self.sentence) - self.next_word > 1 else iter(())

    def after_new_action(self, history_info):
        if not history_info.correct:
            self.is_correct = False
        elif not self.history:
            self.is_correct = True

        self.history.append(history_info.score)
        self.score += history_info.score.score

    def __str__(self):
        return "[{}]".format(self.score) + ", ".join(i.action.__name__ for i in self.history)

    def __repr__(self):
        return str(self)

    def to_conllu_sentence(self):
        def convert_node(node):
            return CoNLLUNode(node.id, node.form, node.lemma, node.cpos,
                       node.pos, node.feats,
                       self.pred_heads[node.id], self.pred_labels[node.id],
                              "_", "_")
        return CoNLLUSentence(convert_node(node) for node in self.sentence if node.id > 0)


class ArcHybridTransition(object):
    pass


class ArcHybridActions(list):
    class ARC_LEFT(ArcHybridTransition):
        require_relation = True
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return not state.is_stack_empty and not state.is_buffer_empty

        @classmethod
        def do_action(cls, state, relation, history_info):
            """
            :type state: ArcHybridState
            """
            child = state.stack.pop()
            parent = state.next_word_root

            state.pred_heads[child.id] = parent.id
            state.pred_labels[child.id] = relation

            bestOp = 0
            hoffset = 1 if state.options.headFlag else 0
            if state.options.rlMostFlag:
                state.leftmost_child_ids[parent.id] = state.leftmost_child_ids[child.id]
            if state.options.rlFlag:
                state.left_child_ids[parent.id] = child.id
            state.after_new_action(history_info)

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            stack_back = state.stack[-1]
            ret = all(h.id != stack_back.parent_id for h in chain(state.s1, state.beta)) and \
                  all(d.parent_id != stack_back.id for d in chain(state.b, state.beta))
            if with_relation:
                ret = (ret and relation == stack_back.relation)
            return ret

        def __str__(self):
            return "ARC_LEFT"

        def __repr__(self):
            return str(self)

    class ARC_RIGHT(ArcHybridTransition):
        require_relation = True
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return len(state.stack) >= 2 and state.stack[-1].id != 0

        @classmethod
        def do_action(cls, state, relation, history_info):
            """
            :type state: ArcHybridState
            """
            child = state.stack.pop()
            parent = state.stack[-1]

            state.pred_heads[child.id] = parent.id
            state.pred_labels[child.id] = relation

            bestOp = 1
            hoffset = 1 if state.options.headFlag else 0
            if state.options.rlMostFlag:
                state.rightmost_child_ids[parent.id] = state.rightmost_child_ids[child.id]
            if state.options.rlFlag:
                state.right_child_ids[parent.id] = child.id
            state.after_new_action(history_info)

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            stack_back = state.stack[-1]
            ret = all(h.id != stack_back.parent_id and \
                      h.parent_id != stack_back.id for h in chain(state.b, state.beta))
            if with_relation:
                ret = (ret and relation == stack_back.relation)
            return ret

        def __str__(self):
            return "ARC_RIGHT"

        def __repr__(self):
            return str(self)

    class SHIFT(ArcHybridTransition):
        require_relation = False
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return not state.is_buffer_empty and state.next_word_root.id != 0

        @classmethod
        def do_action(cls, state, relation, history_info):
            """
            :type state: ArcHybridState
            """
            state.stack.append(state.next_word_root)
            state.next_word += 1
            state.after_new_action(history_info)

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            return all(h.id != state.next_word_root.parent_id for h in chain(state.s1, state.alpha)) and \
            all(d.parent_id != state.next_word_root.id for d in state.stack)

        def __str__(self):
            return "SHIFT"

        def __repr__(self):
            return str(self)

    class INVALID(ArcHybridTransition):
        @staticmethod
        def can_do_action(state):
            return False

        @staticmethod
        def do_action(state, relation):
            """
            :type state: ArcHybridState
            """
            raise RuntimeError("Invalid action.")

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            return False

    class Combo(ArcHybridTransition):
        def __init__(self, seq):
            self.action_seq = list(seq) # type: [ArcHybridTransition]

            require_relation_count = sum(1 for i in self.action_seq if i.require_relation)
            if require_relation_count == 0:
                self.require_relation = False
            elif require_relation_count == 1:
                self.require_relation = True
            else:
                raise NotImplementedError

        def can_do_action(self, state):
            current_state = state # type: ArcHybridState
            for action in self.action_seq:
                if not action.can_do_action(current_state):
                    return False
                current_state = current_state.copy_for_foresee()
                action.do_action(current_state, "XXX")
            return True

        def do_action(self, state, relation):
            for action in self.action_seq:
                action.do_action(state, relation)

        def is_correct(self, state, relation, with_relation=True):
            current_state = state # type: ArcHybridState
            for action in self.action_seq:
                if not action.is_correct(current_state, relation,
                                         with_relation=with_relation):
                    return False
                current_state = current_state.copy_for_foresee()
                action.do_action(current_state, relation)
            return True

        def __str__(self):
            return "Combo({})".format(",".join(str(i) for i in self.action_seq))

        def __repr__(self):
            return self.__str__()

    ActionWithRelation = namedtuple("ActionWithRelation", ["action", "relation",
                                                           "action_index", "relation_index"])

    def __init__(self, relations, action_file):
        if action_file is None:
            actions = [self.ARC_LEFT, self.ARC_RIGHT, self.SHIFT]
        else:
            actions = self.generate_actions(action_file)

        super(ArcHybridActions, self).__init__(actions)

        logger.info("Actions: {}".format(self))
        self.relations = relations
        self.decoded_with_relation = []

        for action_idx, action in enumerate(self):
            if not action.require_relation:
                self.decoded_with_relation.append(
                    self.ActionWithRelation(action, None, action_idx, -1))

        for relation_idx, relation in enumerate(self.relations):
            for action_idx, action in enumerate(self):
                if action.require_relation:
                    self.decoded_with_relation.append(
                        self.ActionWithRelation(action, relation, action_idx, relation_idx))

    @classmethod
    def generate_actions(cls, action_file):
        str_to_action = {"arc_left": cls.ARC_LEFT, "arc_right": cls.ARC_RIGHT,
                         "shift": cls.SHIFT}
        with open(action_file) as f:
            for line in f:
                if not line:
                    continue
                actions = [str_to_action[i] for i in line.strip().split()]
                if len(actions) == 1:
                    yield actions[0]
                else:
                    yield cls.Combo(actions)



class EdgeStatistics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_edge = 0
        self.correct_edge = 0
        self.correct_labeled_edge = 0

    def add_state(self, state):
        for i in state.sentence:
            if i.parent_id is not None:
                self.total_edge += 1
                if state.pred_heads[i.id] == i.parent_id:
                    self.correct_edge += 1
                    if state.pred_labels[i.id] == i.relation:
                        self.correct_labeled_edge += 1

    @property
    def UP(self):
        return self.correct_edge / self.total_edge

    @property
    def LP(self):
        return self.correct_labeled_edge / self.total_edge
