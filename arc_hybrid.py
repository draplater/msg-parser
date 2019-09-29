from __future__ import division

from beam import Beam
import nn
import dynet as dn
from collections import namedtuple
from operator import attrgetter
import transition_utils, time, random


ActionScore = namedtuple("ActionScore", ["relation", "action", "score", "score_repr"])


class ArcHybridLSTM(object):
    def __init__(self, words, pos, rels, w2i, options):
        self.model = dn.Model()
        self.trainer = dn.AdamTrainer(self.model)
        random.seed(1)

        # noinspection PyUnresolvedReferences
        self.activations = {'tanh': dn.tanh, 'sigmoid': dn.logistic, 'relu': dn.rectify,
                            'tanh3': (lambda x: dn.tanh(dn.cwise_multiply(dn.cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.options = options
        self.oracle = options.oracle
        self.ldims = options.lstm_dims * 2
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.relation = options.relation
        if self.relation:
            self.rels = {word: ind for ind, word in enumerate(rels)}
            self.irels = rels
        else:
            self.rels = {"X": 0}
            self.irels = ["X"]

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window

        self.nnvecs = self.headFlag + self.rlFlag * 2 + self.rlMostFlag * 2
        self.actions = transition_utils.ArcHybridActions(self.irels, options.action_file)

        if options.external_embedding is not None:
            self.extrnd, self.elookup, self.edim = nn.get_external_embedding(
                self.model, options.external_embedding)
            print('Load external embedding. Vector dimensions', self.edim)
        else:
            self.extrnd, self.elookup, self.edim = None, None, 0

        dims = self.wdims + self.pdims + self.edim
        self.rnn = nn.BiLSTM(self.model, [dims] + [self.ldims] * options.lstm_layers)

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.word2lstm = self.model.add_parameters((self.ldims, self.wdims + self.pdims + self.edim))
        self.word2lstmbias = self.model.add_parameters((self.ldims))
        # self.lstm2lstm = self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims))
        # self.lstm2lstmbias = self.model.add_parameters((self.ldims))

        input_dims = self.ldims * self.nnvecs * (self.k + 1)
        action_dims = [input_dims, self.hidden_units, self.hidden2_units, len(self.actions)]
        action_dims = [i for i in action_dims if i != 0]
        self.action_classifier = nn.DenseLayers(self.model, action_dims, self.activation)

        relation_dims = [input_dims, self.hidden_units, self.hidden2_units,
                       len(self.actions.decoded_with_relation)]
        relation_dims = [i for i in relation_dims if i != 0]
        self.relation_classifier = nn.DenseLayers(self.model, relation_dims, self.activation)

        if self.options.beam_size == 0:
            self.options.beam_search = False

    def evaluate_all_states(self, states):
        input_tensors = dn.concatenate_to_batch(
            [state.get_input_tensor(self.k, self.empty) for state in states])
        action_outputs = self.action_classifier(input_tensors)
        relation_outputs = None
        if self.relation:
            relation_outputs = self.relation_classifier(input_tensors)
        return action_outputs, relation_outputs

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        evec = self.elookup[1] if self.edim > 0 else None
        paddingWordVec = self.wlookup[1]
        paddingPosVec = self.plookup[1] if self.pdims > 0 else None

        paddingVec = dn.tanh(self.word2lstm.expr() * dn.concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias.expr() )
        self.empty = paddingVec if self.nnvecs == 1 else dn.concatenate([paddingVec for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train):
        input_vecs = []
        ret = [None for i in range(len(sentence) + 1)]

        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))
            word_vec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0] # 100,0
            pos_vec = self.plookup[int(self.pos.get(root.pos, 0))] if self.pdims > 0 else None # 25,0
            ext_vec = self.elookup[int(self.extrnd.get(root.form, self.extrnd.get(root.norm, 0)))] \
                if self.edim > 0 else None
            input_vecs.append(dn.concatenate(filter(None, [word_vec, pos_vec, ext_vec]))) # 225x1

        output_vecs = self.rnn(input_vecs)
        for root, output_vec in zip(sentence, output_vecs):
            ret[root.id] = output_vec

        return ret

    def work(self, sentence, oracle=None, initial_state=None):
        train = oracle is not None

        if initial_state is None:
            initial_state = transition_utils.ArcHybridState(sentence, self.options)
            lstm_outputs = self.getWordEmbeddings(sentence, train)
            initial_state.set_lstm_map(lstm_outputs)

        beam = Beam(maxsize=self.options.beam_size)
        beam.push(initial_state)
        assert not initial_state.history or initial_state.is_correct
        gold_state = None

        while not beam[0].is_finished():
            gold_state = None
            new_beam_candidates = Beam(maxsize=self.options.beam_size)
            gold_candidate = None
            action_scores, label_scores = self.evaluate_all_states(i for i in beam)
            action_scores_np = action_scores.npvalue()
            label_scores_np = label_scores.npvalue()
            if len(beam) == 1:
                action_scores_np = action_scores_np.reshape(
                    action_scores_np.shape + (1,))
                label_scores_np = label_scores_np.reshape(
                    label_scores_np.shape + (1,))

            for state_idx, state in enumerate(beam):
                can_do_action = [i.can_do_action(state) for i in self.actions]
                for joint_idx, (action, relation, action_idx, relation_idx) \
                        in enumerate(self.actions.decoded_with_relation):
                    if not can_do_action[action_idx]:
                        continue
                    local_score = action_scores_np[action_idx, state_idx] + label_scores_np[joint_idx, state_idx]
                    if train and state.is_correct and (action, relation) == oracle[len(state.history)] and self.options.loss_aug:
                        local_score -= self.options.loss_aug
                    candidate = transition_utils.StateCandidate(
                            state_idx, joint_idx, state.score + local_score, local_score)
                    if train and state.is_correct and (action, relation) == oracle[len(state.history)]:
                        gold_candidate = candidate
                    new_beam_candidates.push(candidate)

            def candidate_to_state(candidate_):
                state = beam[candidate_.state_idx]
                new_state = state.copy()
                action, relation, action_idx, relation_idx = self.actions.decoded_with_relation[candidate_.joint_idx]
                correctness = (action, relation) == oracle[len(state.history)] if train else True
                score_repr = dn.pick_batch_elem(action_scores, candidate_.state_idx)[action_idx] + \
                             dn.pick_batch_elem(label_scores, candidate_.state_idx)[candidate_.joint_idx] \
                    if train else None
                action.do_action(
                    new_state, relation, transition_utils.History(
                      ActionScore(relation, action, candidate_.local_score, score_repr), correctness))
                return new_state

            new_beam = Beam(maxsize=self.options.beam_size, key=attrgetter("score"))
            for candidate in new_beam_candidates:
                new_state = candidate_to_state(candidate)
                new_beam.push(new_state)
                if train and new_state.is_correct:
                    state = beam[candidate.state_idx]
                    assert not state.history or state.is_correct
                    gold_state = new_state

            if train and gold_state is None:
                gold_state = candidate_to_state(gold_candidate)
                beam = new_beam
                break  # early update

            beam = new_beam

        state_to_update = max((i for i in beam), key=attrgetter("score"))
        return state_to_update, gold_state

    def work_dynamic_oracle(self, sentence, train):
        state = transition_utils.ArcHybridState(sentence, self.options)
        lstm_outputs = self.getWordEmbeddings(sentence, train)
        state.set_lstm_map(lstm_outputs)
        errs = []
        while not state.is_finished():
            can_do_action = [i.can_do_action(state) for i in self.actions]
            is_correct = [i.is_correct(state, None, False) if can_do else False
                          for i, can_do in zip(self.actions, can_do_action)]
            action_scores, label_scores = self.evaluate_all_states([state])
            action_scores_np = action_scores.npvalue()
            label_scores_np = label_scores.npvalue()
            best_valid = None
            best_valid_score = None
            best_wrong = None
            best_wrong_score = None
            best = None
            best_score = None
            for joint_idx, (action, relation, action_idx, relation_idx) \
                   in enumerate(self.actions.decoded_with_relation):
                if not can_do_action[action_idx]:
                    continue
                local_score = action_scores_np[action_idx] + label_scores_np[joint_idx]
                if train:
                    joint_correct = is_correct[action_idx]
                    if action.require_relation:
                        joint_correct = joint_correct and relation == state.stack[-1].relation
                    if joint_correct and (best_valid is None or local_score > best_valid_score):
                        best_valid = joint_idx
                        best_valid_score = local_score
                    if not joint_correct and (best_wrong is None or local_score > best_wrong_score):
                        best_wrong = joint_idx
                        best_wrong_score = local_score
                else:
                    if best is None or local_score > best_score:
                        best = joint_idx
                        best_score = local_score

            if train:
                best = best_valid
                best_score = best_valid_score
                if best_wrong is not None and best_valid != best_wrong and \
                                        best_valid_score - best_wrong_score <= 1.0 and \
                        (best_valid_score <= best_wrong_score or random.random() < 0.1):
                    best = best_wrong
                    best_score = best_wrong_score

            best_info = self.actions.decoded_with_relation[best]
            best_info.action.do_action(state, best_info.relation, transition_utils.History(
                      ActionScore(best_info.relation, best_info.action,
                                  best_score, None), True))

            if train and best_wrong is not None and best_valid_score < best_wrong_score + 1.0:
                def get_score_repr(idx):
                    joint_info = self.actions.decoded_with_relation[idx]
                    return action_scores[joint_info.action_index] + label_scores[idx]
                loss = get_score_repr(best_wrong) - get_score_repr(best_valid)
                errs.append(loss)

        return state, errs

    def Predict(self, sentences):
        for iSentence, sentence in enumerate(sentences):
            if iSentence % 100 == 0:
                print("Predicting sentence {}.".format(iSentence))
            self.Init()
            if self.options.beam_search:
                final_state, _ = self.work(sentence)
            else:
                final_state, _ = self.work_dynamic_oracle(sentence, False)
            yield final_state.to_conllu_sentence()
            dn.renew_cg()

    def Train(self, sentences):
        random.shuffle(sentences)
        self.Init()

        edge_statistics = transition_utils.EdgeStatistics()
        total_loss = 0.0
        errs = []
        start = time.time()

        for sent_idx, sentence in enumerate(sentences):
            if sent_idx % 100 == 0:
                if sent_idx != 0:
                    print('Processing sentence number:', sent_idx,
                          'Loss: %.2f' % total_loss,
                          'UP: %.2f' % (edge_statistics.UP * 100),
                          'LP: %.2f' % (edge_statistics.LP * 100),
                          'Time', time.time() - start)
                # gc.collect()
                # objs = Counter(type(i).__name__ for i in gc.get_objects()).most_common(5)
                # print(objs)
                start = time.time()
                edge_statistics.reset()
                total_loss = 0.0

            if self.options.beam_search:
                gold_state = None
                while gold_state is None or not gold_state.is_finished():
                    oracle = self.get_orcale(sentence)
                    if not oracle:
                        print("Cannot get oracle for sentence {}.".format(sent_idx))
                        break
                    state_to_update, gold_state = self.work(sentence, oracle, gold_state)

                    if state_to_update is not gold_state:
                        total_score = sum(i.score for i in state_to_update.history)
                        total_gold_score = sum(i.score for i in gold_state.history)
                        assert total_score > total_gold_score
                        edge_statistics.add_state(state_to_update)

                        need_to_update = False
                        for best_score, gold_score in zip(state_to_update.history, gold_state.history):
                            if (best_score.action, best_score.relation) != \
                                   (gold_score.action, gold_score.relation):
                                need_to_update = True
                            if need_to_update:
                                errs.append(best_score.score_repr - gold_score.score_repr)
                    if not self.options.full_search:
                        break
            else:
                oracle = self.get_orcale(sentence)
                if not oracle:
                    continue
                state, errs = self.work_dynamic_oracle(sentence, True)
                edge_statistics.add_state(state)

            if len(errs):
                eerrs = dn.esum(errs)
                total_loss += eerrs.scalar_value()
                eerrs.backward()
                self.trainer.update()
                errs = []

                dn.renew_cg()
                self.Init()

        self.trainer.update_epoch()

    def get_orcale(self, conll_sentence):
        test_state = transition_utils.ArcHybridState(conll_sentence, self.options)
        test_state.set_lstm_map([None for i in range(len(conll_sentence))])
        oracle = []
        while not test_state.is_finished():
            has_action = False
            for action in self.actions:
                if action.can_do_action(test_state) and action.is_correct(test_state, None, False):
                    relation = None
                    if action.require_relation:
                        relation = test_state.stack[-1].relation
                    oracle.append((action, relation))
                    score = ActionScore(relation, action, 1, None)
                    action.do_action(test_state, relation, transition_utils.History(score, score))
                    has_action = True
                    break
            if not has_action:
                return None
        for node in test_state.sentence:
            if node.id > 0:
                assert node.parent_id == test_state.pred_heads[node.id]
        return oracle
        # while not test_state.is_finished():
        #     if not test_state.is_stack_empty and test_state.next_word < len(test_state.sentence) and test_state.stack[-1].parent_id == test_state.next_word_root.id:
        #         action = transition_utils.ArcHybridActions.ARC_LEFT
        #         relation = test_state.stack[-1].relation
        #     elif len(test_state.stack) >= 2 and test_state.stack[-1].parent_id == test_state.stack[-2].id:
        #         action = transition_utils.ArcHybridActions.ARC_RIGHT
        #         relation = test_state.stack[-1].relation
        #     else:
        #         action = transition_utils.ArcHybridActions.SHIFT
        #         relation = None
        #     oracle.append((action, relation))
        #     score = ActionScore(relation, action, 1, None)
        #     action.do_action(test_state, relation, transition_utils.History(score, score))
        # return oracle
