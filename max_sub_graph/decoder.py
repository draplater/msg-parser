# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/
import numpy as np
from .cffi_builder import build_1ec2p, build_1ec2p_vine


class oneec2p(object):
    ffi = None
    lib = None

    def __init__(self, options):
        if not self.ffi:
            self.lib, self.ffi = build_1ec2p()

    def __call__(self, scores):
        sent_len = scores.shape[0]
        new_matrix = np.zeros((sent_len + 1, 256, 2), dtype=np.float64)
        new_matrix[1:sent_len+1, 1:sent_len+1, 0] = scores
        arcs = np.zeros((sent_len * sent_len, 2), dtype=np.int32)
        score_pointer = self.ffi.cast("double score[{}][256][2]".format(sent_len), new_matrix.ctypes.data)
        arcs_pointer = self.ffi.cast("int arcs[{}][2]".format(sent_len * sent_len), arcs.ctypes.data)
        count = self.lib.parse(sent_len, score_pointer, arcs_pointer)
        ret = np.zeros(scores.shape, dtype=bool)
        for i in range(count):
            ret[arcs[i][0] - 1][arcs[i][1] - 1] = True
        return ret


class oneec2p_vine(object):
    ffi = None
    lib = None

    def __init__(self, options):
        self.length = options.vine_arc_length
        if not self.ffi:
            self.lib, self.ffi = build_1ec2p_vine()

    def __call__(self, scores):
        sent_len = scores.shape[0]
        new_matrix = np.zeros((sent_len + 1, 256, 2), dtype=np.float64)
        new_matrix[1:sent_len+1, 1:sent_len+1, 0] = scores
        arcs = np.zeros((sent_len * sent_len, 2), dtype=np.int32)
        score_pointer = self.ffi.cast("double score[{}][256][2]".format(sent_len), new_matrix.ctypes.data)
        arcs_pointer = self.ffi.cast("int arcs[{}][2]".format(sent_len * sent_len), arcs.ctypes.data)
        count = self.lib.parse_vine(self.length, sent_len, score_pointer, arcs_pointer)
        ret = np.zeros(scores.shape, dtype=bool)
        for i in range(count):
            ret[arcs[i][0] - 1][arcs[i][1] - 1] = True
        return ret


class arcfactor(object):
    def __init__(self, options):
        pass

    def __call__(self, scores):
        ret = np.zeros(scores.shape, dtype=bool)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i][j] > 0:
                    ret[i][j] = True
        return ret

