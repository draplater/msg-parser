import random
import time
from operator import itemgetter

import dynet as dn

import graph_utils
import nn
from logger import logger
from max_sub_graph import cost_augments, decoder
from nn import BiLinear, DenseLayers


class MaxSubGraphLSTM(object):

    activations = {'tanh': dn.tanh, 'sigmoid': dn.logistic, 'relu': dn.rectify,
                   'tanh3': (lambda x: dn.tanh(dn.cwise_multiply(
                    dn.cwise_multiply(x, x), x)))}
    decoders = {"arcfactor": decoder.arcfactor, "1ec2p": decoder.oneec2p, "1ec2p-vine": decoder.oneec2p_vine}

    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = dn.Model()
        random.seed(1)
        self.trainer = dn.AdamTrainer(self.model)

        self.activation = self.activations[options.activation]
        self.decoder = self.decoders[options.decoder](options)
        self.test_decoder = self.decoders[options.test_decoder](options) \
            if options.test_decoder is not None \
            else self.decoder
        self.cost_augment = cost_augments[options.cost_augment]

        self.labelsFlag = options.labelsFlag
        self.options = options

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}  # type: dict[str, int]
        self.irels = rels

        if options.external_embedding is not None:
            self.extrnd, self.elookup, self.edim = nn.get_external_embedding(
                self.model, options.external_embedding)
            logger.info('Load external embedding. Vector dimensions %d', self.edim)
        else:
            self.extrnd, self.elookup, self.edim = None, None, 0

        dims = self.wdims + self.pdims + self.edim
        self.rnn = nn.BiLSTM(self.model, [dims] + [self.ldims * 2] * options.lstm_layers)

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))


        if self.hidden2_units > 0:
            dense_dims = [self.hidden_units, self.hidden2_units, 1]
            use_bias = [True, False]
        else:
            dense_dims = [self.hidden_units, 1]
            # use_bias = [dn.NormalInitializer(0, 0)]
            use_bias = [False]

        self.head_dense_layer = DenseLayers(self.model,
                                            [self.ldims * 2, self.hidden_units], self.activation)
        self.dep_dense_layer = DenseLayers(self.model,
                                           [self.ldims * 2, self.hidden_units], self.activation)

        self.fusion_layer = nn.Biaffine(self.model, self.hidden_units, self.activation)

        if self.labelsFlag:
            self.relation_binear_layer = BiLinear(self.model, self.ldims * 2, self.hidden_units)
            relation_dense_dims = list(dense_dims)
            relation_dense_dims[-1] = len(self.irels)

            self.relation_dense_layer = DenseLayers(self.model, relation_dense_dims,
                                                    self.activation)

    def get_vecs(self, node):
            wordvec = self.wlookup[int(self.vocab.get(node.norm, 0))] if self.wdims > 0 else None
            posvec = self.plookup[int(self.pos.get(node.postag, 0))] if self.pdims > 0 else None
            evec = self.elookup[int(self.extrnd.get(node.form, self.extrnd.get(node.norm, 0)))]\
                if self.edim > 0 else None
            return dn.concatenate(filter(None, [wordvec, posvec, evec]))

    def __evaluate(self, lstm_output):
        length = len(lstm_output)

        # (i, j) -> (i * length + j,)
        # i = k / length, j = k % length
        # 1 1 2 2 3 3 4 4 ..
        heads = [dn.transpose(self.activation(self.head_dense_layer(lstm_output[i]))) for i in range(length)]
        mods = [self.activation(self.dep_dense_layer(lstm_output[i])) for i in range(length)]
        head_part = dn.concatenate_to_batch([heads[i // len(lstm_output)] for i in range(length * length)])
        # 1 2 3 4 .. 1 2 3 4 ...
        mod_part = dn.concatenate_to_batch([mods[i] for i in range(length)] * length)

        output = self.fusion_layer(head_part, mod_part)

        exprs = [[dn.pick_batch_elem(output, i * length + j) for j in range(length)] for i in range(length)]
        scores = output.npvalue()
        scores = scores.reshape((len(lstm_output), len(lstm_output)))

        return scores, exprs

    def __evaluate_labels(self, lstm_output, edges):
        """
        :type lstm_output: list[dn.Expression]
        :type edges: Edge
        :return: 
        """
        rheadfov = [None] * len(lstm_output)
        rmodfov = [None] * len(lstm_output)

        for source, label, target in edges:
            if rheadfov[source] is None:
                rheadfov[source] = self.relation_binear_layer.w1.expr() * lstm_output[source]
            if rmodfov[target] is None:
                rmodfov[target] = self.relation_binear_layer.w2.expr() * lstm_output[target]

            hidden = self.activation(
                rheadfov[source] + rmodfov[target] +
                self.relation_binear_layer.bias.expr())
            output = self.relation_dense_layer(hidden)

            yield output

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def Predict(self, graphs):
        for iSentence, sentence in enumerate(graphs):
            vecs = [self.get_vecs(i) for i in sentence]
            lstm_output = self.rnn([vecs[i] for i in range(len(sentence))])
            scores, exprs = self.__evaluate(lstm_output)

            output_graph = self.test_decoder(scores)

            edges = []
            for source_id in range(len(sentence)):
                for target_id in range(len(sentence)):
                    if output_graph[source_id][target_id]:
                        edges.append(graph_utils.Edge(source_id, "X", target_id))

            if self.labelsFlag:
                labeled_edges = []

                for edge, r_scores_expr in \
                        zip(edges, self.__evaluate_labels(lstm_output, edges)):
                    r_scores = r_scores_expr.value()
                    label_index = max(((l, scr) for l, scr in enumerate(r_scores)), key=itemgetter(1))[0]
                    label = self.irels[label_index]
                    labeled_edges.append(graph_utils.Edge(edge.source, label, edge.target))
                edges = labeled_edges

            dn.renew_cg()
            yield sentence.replaced_edges(edges)

    def Train(self, graphs):
        """
        :type graphs: [Graph]
        :return: 
        """
        eloss = 0.0
        mloss = 0.0
        total_gold_edge = 0
        total_predict_edge = 0
        recalled_gold_edge = 0.0
        correct_predict_edge = 0.0
        start = time.time()

        shuffled_index = range(len(graphs))
        random.shuffle(shuffled_index)

        iSentence = -1
        for g_idx in shuffled_index:
            sentence = graphs[g_idx]  # type: graph_utils.Graph
            dn.renew_cg()
            iSentence += 1
            if iSentence % 100 == 0 and iSentence != 0:
                logger.info(
                    'Processing sentence number: %d, Loss: %.2f, '
                    'Accuracy: %.2f, Recall: %.2f, Time: %.2f',
                    iSentence, eloss,
                    correct_predict_edge / total_predict_edge * 100,
                    recalled_gold_edge / total_gold_edge * 100,
                    time.time()-start
                )
                start = time.time()
                eloss = 0.0
                total_gold_edge = 0
                total_predict_edge = 0
                recalled_gold_edge = 0.0
                correct_predict_edge = 0.0

            vecs = [self.get_vecs(i) for i in sentence]
            lstm_output = self.rnn([vecs[i] for i in range(len(sentence))])
            scores, exprs = self.__evaluate(lstm_output)

            self.cost_augment(scores, sentence, self.options)

            output_graph = self.decoder(scores)
            gold_graph = sentence.to_matrix()

            lerrs = []
            if self.labelsFlag:
                edges = list(sentence.generate_edges())
                for edge, r_scores_expr \
                        in zip(edges, self.__evaluate_labels(lstm_output, edges)):
                    head, label, modifier = edge
                    r_scores = r_scores_expr.value()
                    gold_label_index = self.rels[label]
                    wrong_label_index = max(((l, scr)
                                             for l, scr in enumerate(r_scores)
                                             if l != gold_label_index), key=itemgetter(1))[0]
                    if r_scores[gold_label_index] < r_scores[wrong_label_index] + 1:
                        lerrs.append(
                            r_scores_expr[wrong_label_index] -
                            r_scores_expr[gold_label_index])

            errs = []
            for source_id in range(len(sentence)):
                for target_id in range(len(sentence)):
                    gold_exist = gold_graph[source_id][target_id]
                    output_exist = output_graph[source_id][target_id]
                    if gold_exist and output_exist:
                        total_gold_edge += 1
                        total_predict_edge += 1
                        correct_predict_edge += 1
                        recalled_gold_edge += 1
                    elif not gold_exist and not output_exist:
                        pass
                    elif gold_exist and not output_exist:
                        total_gold_edge += 1
                        errs.append(-exprs[source_id][target_id] + 1)
                    elif not gold_exist and output_exist:
                        total_predict_edge += 1
                        errs.append(exprs[source_id][target_id])
                    else:
                        raise SystemError()

            if len(errs) > 0 or len(lerrs) > 0:
                loss = dn.scalarInput(0.0)
                if len(lerrs):
                    loss += dn.esum(lerrs)
                if len(errs):
                    loss += dn.esum(errs)
                eloss += loss.scalar_value()
                loss.backward()
                self.trainer.update()

        self.trainer.update_epoch()
        logger.info("Loss: %.2f", mloss/iSentence)

