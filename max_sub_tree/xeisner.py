import os
import ctypes
import platform
import random

import subprocess
import numpy as np

from common_utils import under_construction, ensure_dir
from max_sub_tree.empty_node import EdgeToEmpty

def get_lib():
    hostname = platform.node()
    source_dir = os.path.join(os.path.dirname(__file__), "../libs", "xEisner")

    build_dir = os.path.join(source_dir, "build-{}".format(hostname))
    ensure_dir(build_dir)
    lib_path = os.path.join(build_dir, "libxEisner.so")
    print("Building xEisner...")
    p = subprocess.Popen("MAX_SENTENCE_SIZE=128 cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j4",
                         shell=True, cwd=build_dir)
    p.communicate()
    assert p.returncode == 0
    assert os.path.exists(lib_path)
    return ctypes.cdll.LoadLibrary(lib_path)


lib = get_lib()

MAX_SENT_SIZE_NOEMPTY = 256

MAX_SENT_SIZE_EMPTY = 128
MAX_SENT_BITS = 7


def decode_target(head, mod, N):
    if mod >= (1 << MAX_SENT_BITS):
        empty_id = mod >> MAX_SENT_BITS
        position = (mod & ((1 << MAX_SENT_BITS) - 1))
        if position == N + 1:
            position = 0
        return EdgeToEmpty(int(head), int(empty_id), int(position))
    else:
        mod += 1
        assert mod <= N
        assert head <= N
        return int(mod)


def arcs_to_biarcs(gold):
    N = len(gold) - 1
    edges = np.zeros((N, 2), dtype=np.int32)
    for mod, head in enumerate(gold[1:], 1):
        edges[mod - 1, 0] = head - 1
        edges[mod - 1, 1] = mod - 1
    edges_p = edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bi_edges = np.zeros((N, 3), dtype=np.int32)
    bi_edges_p = bi_edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ret = lib.arcs_to_biarcs(N, edges_p, bi_edges_p)
    assert ret == N
    biarcs2 = set()
    biarcs3 = set()
    for i, j, k in bi_edges:
        if i == N:
            i = 0
        else:
            i += 1
        k += 1
        if j == -1:
            biarcs2.add((i, k))
        else:
            j += 1
            biarcs3.add((i, j, k))
    return biarcs2, biarcs3


def arcs_with_empty_to_biarcs(gold, emptys):
    N = len(gold) - 1
    edges = np.zeros((N + len(emptys), 2), dtype=np.int32)
    for mod, head in enumerate(gold[1:], 1):
        edges[mod - 1, 0] = head - 1
        edges[mod - 1, 1] = mod - 1
    for i, (head, id_, position) in enumerate(emptys):
        p = (((id_) << MAX_SENT_BITS) | (position))
        edges[N + i, 0] = head - 1
        edges[N + i, 1] = p

    edges_p = edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bi_edges = np.zeros((N + len(emptys), 3), dtype=np.int32)
    bi_edges_p = bi_edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ret = lib.arcs_with_empty_to_biarcs(N + len(emptys), edges_p, bi_edges_p)
    assert ret == N + len(emptys)
    biarcs2 = set()
    biarcs3 = set()
    for i, j, k in bi_edges:
        if i == N:
            i = 0
        else:
            i += 1
        k = decode_target(i, k, len(gold) - 1)
        if j == -1:
            biarcs2.add((int(i), k))
        else:
            j = decode_target(i, j, len(gold) - 1)
            biarcs3.add((int(i), j, k))
    return biarcs2, biarcs3


def eisner2nd_decode(scores_n, scores2nd2_n, scores2nd3_n):
    N = scores_n.shape[0] - 1
    result = np.zeros((N, 2), dtype=np.int32)
    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_p = scores_n.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    scores2nd2_p = scores2nd2_n.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    scores2nd3_p = scores2nd3_n.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    edge_count = lib.eisner2nd_decode2nd(N, result_p, scores_p, scores2nd2_p, scores2nd3_p)

    heads = np.zeros((N + 1,), dtype=np.int32)
    heads[0] = -1

    for i in range(edge_count):
        mod = result[i, 1] + 1
        head = result[i, 0] + 1
        heads[mod] = head

    return heads


def eisner1st_decode(scores_n):
    N = scores_n.shape[0] - 1
    result = np.zeros((N, 2), dtype=np.int32)
    score_native = np.zeros((N + 2, MAX_SENT_SIZE_NOEMPTY), dtype=np.float64)
    score_native[0:N, 0:N] = scores_n[1:N + 1, 1:N + 1]
    score_native[N, 0:N] = scores_n[0, 1:N + 1]
    score_native[0:N, N] = scores_n[1:N + 1, 0]
    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_p = score_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    edge_count = lib.eisner2nd_decode(N, result_p, scores_p)

    heads = np.zeros((N+1,), dtype=np.int32)
    heads[0] = -1

    for i in range(edge_count):
        mod = result[i, 1] + 1
        head = result[i, 0] + 1
        heads[mod] = head

    return heads


def emptyeisner1st_decode(N, empty_count, scores_n):
    MAX_SENT_BITS = 7
    result = np.zeros((4 * N + 10, 2), dtype=np.int32)
    score_native = np.zeros((empty_count + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    # score_native[:, 0:N+1, 0:N+1] = scores_n[:, 0:N+1, 0:N+1]
    score_native[:, 0:N, 0:N] = scores_n[:, 1:N + 1, 1:N + 1]
    score_native[:, N, 0:N] = scores_n[:, 0, 1:N + 1]
    score_native[:, 0:N, N] = scores_n[:, 1:N + 1, 0]
    score_native[:, N, N] = scores_n[:, 0, 0]
    score_native[1, 0:N, :N+1] = scores_n[1, 1:N+1, :N+1]
    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_p = score_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    edges_count = lib.emptyeisner1st_decode(N, empty_count, result_p, scores_p)
    assert edges_count <= 4 * N + 10

    heads = np.zeros((N+1,), dtype=np.int32)
    emptys = set()
    heads[0] = -1

    for i in range(edges_count):
        head = result[i, 0] + 1
        mod = result[i, 1]
        if mod >= (1 << MAX_SENT_BITS):
            empty_id = mod >> MAX_SENT_BITS
            position = (mod & ((1 << MAX_SENT_BITS) - 1))
            if position == N + 1:
                position = 0
            emptys.add(EdgeToEmpty(head, empty_id, position))
        else:
            mod += 1
            assert mod <= N
            assert head <= N
            heads[mod] = head

    return heads, emptys


def emptyeisner2nd_decode(N, empty_count, scores_n, scores2nd2_n, scores_solid, scores_mid, scores_out):
    result = np.zeros((N * N + N + 10, 2), dtype=np.int32)
    score_native = np.zeros((empty_count + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    score_native[:, 0:N, 0:N] = scores_n[:, 1:N + 1, 1:N + 1]
    score_native[:, N, 0:N] = scores_n[:, 0, 1:N + 1]
    score_native[:, 0:N, N] = scores_n[:, 1:N + 1, 0]
    score_native[1, 0:N, :N+1] = scores_n[1, 1:N+1, :N+1]

    score2nd2_native = np.zeros((empty_count + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    score2nd2_native[:, 0:N, 0:N] = scores2nd2_n[:, 1:N + 1, 1:N + 1]
    score2nd2_native[:, N, 0:N] = scores2nd2_n[:, 0, 1:N + 1]
    score2nd2_native[:, 0:N, N] = scores2nd2_n[:, 1:N + 1, 0]
    score2nd2_native[1, 0:N, :N+1] = scores2nd2_n[1, 1:N+1, :N+1]

    solid_native = np.zeros((N + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    solid_native[0:N, 0:N, 0:N] = scores_solid[1:N + 1, 1:N + 1, 1:N + 1]
    solid_native[N, 0:N, 0:N] = scores_solid[0, 1:N + 1, 1:N + 1]
    solid_native[0:N, 0:N, N] = scores_solid[1:N + 1, 1:N + 1, 0]
    solid_native[0:N, N, 0:N] = scores_solid[1:N + 1, 0, 1:N + 1]

    mid_native = np.zeros((N + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    mid_native[0:N, 0:N, 0:N] = scores_mid[1:N + 1, 0:N, 1:N + 1]
    mid_native[N, 0:N, 0:N] = scores_mid[0, 0:N, 1:N + 1]
    mid_native[0:N, 0:N, N] = scores_mid[1:N + 1, 0:N, 0]

    out_native = np.zeros((N + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    out_native[0:N, 0:N, 0:N] = scores_out[1:N + 1, 1:N + 1, 0:N]
    out_native[N, 0:N, 0:N] = scores_out[0, 1:N + 1, 0:N]
    out_native[0:N, N, 0:N] = scores_out[1:N + 1, 0, 0:N]

    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_p = score_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    scores2nd2_p = score2nd2_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    solid_p = solid_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mid_p = mid_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_p = out_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    edge_count = lib.emptyeisner2nd_decode(N, empty_count, result_p, scores_p, scores2nd2_p,
                                           solid_p, mid_p, out_p)

    heads = np.zeros((N + 1,), dtype=np.int32)
    heads[0] = -1
    emptys = set()

    for i in range(edge_count):
        head = result[i, 0] + 1
        mod = result[i, 1]
        if mod >= (1 << MAX_SENT_BITS):
            empty_id = mod >> MAX_SENT_BITS
            position = (mod & ((1 << MAX_SENT_BITS) - 1))
            if position == N + 1:
                position = 0
            emptys.add(EdgeToEmpty(head, empty_id, position))
        else:
            mod += 1
            assert mod <= N
            assert head <= N
            heads[mod] = head

    return heads, emptys


@under_construction
def emptyeisner1stf_decode(N, empty_count, scores_n):
    MAX_SENT_BITS = 7
    result = np.zeros((4 * N + 10, 2), dtype=np.int32)
    score_native = np.zeros((empty_count + 1, MAX_SENT_SIZE_EMPTY, MAX_SENT_SIZE_EMPTY), dtype=np.float64)
    score_native[:, 0:N, 0:N] = scores_n[:, 1:N + 1, 1:N + 1]
    score_native[:, N, 0:N] = scores_n[:, 0, 1:N + 1]
    score_native[:, 0:N, N] = scores_n[:, 1:N + 1, 0]
    score_native[:, N, N] = scores_n[:, 0, 0]
    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_p = score_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    edges_count = lib.emptyeisner1stf_decode(N, empty_count, result_p, scores_p)
    assert edges_count <= 4 * N + 10

    heads = np.zeros((N+1,), dtype=np.int32)
    emptys = set()
    heads[0] = -1

    for i in range(edges_count):
        head = result[i, 0] + 1
        mod = result[i, 1]
        if mod >= (1 << MAX_SENT_BITS):
            empty_id = mod >> MAX_SENT_BITS
            position = (mod & ((1 << MAX_SENT_BITS) - 1)) + 1
            if position == N + 1:
                position = 0
            emptys.add(EdgeToEmpty(head, empty_id, position))
        else:
            mod += 1
            assert mod <= N
            assert head <= N
            heads[mod] = head

    return heads, emptys


def test_0():
    scores = np.array(
        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 1],
         [1, 0, 0, 1]]
    )
    scores_empty = np.array(
        [[-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -1, -1, -1]]
    )
    print(eisner1st_decode(scores))
    print(emptyeisner1st_decode(3, 1, np.stack([scores, scores_empty])))


def test():
    scores = np.array(
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]
    )
    scores_empty = np.array(
        [[-1, -1, -1, -1],
         [-1, 100, -1, 100],
         [-1, -1, 100, -1],
         [-1, -1, -1, 100]]
    )
    print(emptyeisner1stf_decode(3, 1, np.stack([scores, scores_empty])))


def test_real_data():
    from max_sub_tree.mstlstm_empty import SentenceWithEmpty
    sents = SentenceWithEmpty.from_file("/home/chenyufei/Development/large-data/emptynode-data/cn.trn.ept")
    r = random.Random(42)
    r.shuffle(sents)
    for sent in sents:
        gold_heads = np.array([i.parent_id for i in sent], dtype=np.int32)
        scores = -0.01 * np.ones((len(sent), len(sent)))
        for mod, head in enumerate(gold_heads[1:], 1):
            scores[head][mod] += 1

        heads_1 = eisner1st_decode(scores)
        if not np.array_equal(gold_heads, heads_1):
            print(gold_heads)
            print(heads_1)
            raise AssertionError

        emptys_padding = -10000 * np.ones((len(sent), len(sent)))
        heads_2, emptys = emptyeisner1st_decode(len(sent) - 1, 1, np.stack([scores, emptys_padding]))
        if not np.array_equal(gold_heads, heads_2):
            print(gold_heads)
            print(heads_2)
            raise AssertionError

        scores_empty = -0.01 * np.ones((len(sent), len(sent)))
        for head, id_, position in sent.empty_nodes:
            scores_empty[head][position] += 1
        heads_2, emptys = emptyeisner1st_decode(len(sent) - 1, 1, np.stack([scores, scores_empty]))
        if not np.array_equal(gold_heads, heads_2):
            print(gold_heads)
            print(heads_2)
            print(sent.empty_nodes)
            print(emptys)
            raise AssertionError

        if not np.array_equal(sent.empty_nodes, emptys):
            print(gold_heads)
            print(heads_2)
            print(sent.empty_nodes)
            print(emptys)
            raise AssertionError

        # heads, emptys = emptyeisner1stf_decode(len(sent) - 1, 1, np.stack([scores, scores_empty]))
        # print(sent.empty_nodes, emptys)
        # assert sent.empty_nodes == emptys


def test_real_data_2nd():
    from max_sub_tree.mstlstm_empty import SentenceWithEmpty
    sents = SentenceWithEmpty.from_file("/home/chenyufei/Development/large-data/emptynode-data/cn.trn.small.ept")
    r = random.Random(46)
    r.shuffle(sents)
    for sent in sents:
        gold_heads = np.array([i.parent_id for i in sent], dtype=np.int32)
        bi_arcs2, bi_arcs3 = arcs_with_empty_to_biarcs(gold_heads, sent.empty_nodes)
        # scores = -0.01 * np.ones((2, len(sent), len(sent)))
        scores = np.zeros((2, len(sent), len(sent)))
        # for mod, head in enumerate(gold_heads[1:], 1):
        #     scores[0, head, mod] += 1
        #
        # for head, id_, position in sent.empty_nodes:
        #     scores[1, head, position] += 1

        scores2 = -0.01 * np.ones((2, len(sent), len(sent)))
        for s,t in bi_arcs2:
            if isinstance(t, EdgeToEmpty):
                scores2[1, s, t.position] += 1
            else:
                scores2[0, s, t] += 1

        # scores_solid = np.zeros((len(sent), len(sent), len(sent)))
        # scores_mid = np.zeros((len(sent), len(sent), len(sent)))
        # scores_out = np.zeros((len(sent), len(sent), len(sent)))
        scores_solid = -0.01 * np.ones((len(sent), len(sent), len(sent)))
        scores_mid = -0.01 * np.ones((len(sent), len(sent), len(sent)))
        scores_out = -0.01 * np.ones((len(sent), len(sent), len(sent)))
        for s, m, t in bi_arcs3:
            if isinstance(m, int) and isinstance(t, int):
                scores_solid[s, m, t] += 1
            elif isinstance(m, EdgeToEmpty) and isinstance(t, int):
                scores_mid[s, m.position, t] += 1
            elif isinstance(m, int) and isinstance(t, EdgeToEmpty):
                scores_out[s, m, t.position] += 1
            else:
                raise TypeError
        heads, emptys = emptyeisner2nd_decode(len(sent) - 1, 1, scores, scores2, scores_solid, scores_mid, scores_out)
        assert np.array_equal(heads, gold_heads)
        assert sent.empty_nodes == emptys



if __name__ == '__main__':
    test_real_data_2nd()