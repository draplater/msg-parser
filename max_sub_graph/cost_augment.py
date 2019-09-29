import numpy as np


class Disable(object):
    def __init__(self, options):
        pass

    def __call__(self, scores, gold_graph):
        pass

    def get_loss_shift(self, output_matrix, gold_matrix):
        return 0.0


class Basic(object):
    def __init__(self, options):
        self.decrease = options.basic_costaug_decrease

    def __call__(self, scores, gold_graph):
        for source, label, target in gold_graph.generate_edges():
            scores[source][target] -= self.decrease

    def get_loss_shift(self, output_matrix, gold_matrix):
        loss_shift = np.sum(gold_matrix) * self.decrease
        correct_matrix = np.logical_and(output_matrix, gold_matrix)
        loss_shift -= np.sum(correct_matrix) * self.decrease
        return loss_shift


class Hamming(object):
    def __init__(self, options):
        self.a = options.hamming_a
        self.b = options.hamming_b

    def __call__(self, scores, gold_graph):
        """
        :param scores:
        :type gold_graph: graph_utils.Graph
        :rtype: None
        """
        for source in range(len(gold_graph)):
            for target in range(len(gold_graph)):
                scores[source, target] += self.a

        for source, label, target in gold_graph.generate_edges():
            scores[source, target] -= self.a + self.b

    def get_loss_shift(self, output_matrix, gold_matrix):
        loss_shift = np.sum(gold_matrix) * self.b
        correct_matrix = np.logical_and(output_matrix, gold_matrix)
        wrong_matrix = output_matrix - correct_matrix
        loss_shift += np.sum(wrong_matrix) * self.a
        loss_shift -= np.sum(correct_matrix) * self.b
        return loss_shift


cost_augmentors = {"basic": Basic,
                   "hamming": Hamming,
                   "disable": Disable}
