# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

from six.moves import range
import itertools

import numpy as np


min_float = -10000000.0


def eisner2nd(scores, scores2nd2, scores2nd3, gold=None):
    from max_sub_tree import xeisner
    N = scores.shape[0] - 1
    # The score is not used anymore
    # scores_n = np.copy(scores)
    # scores2nd2_n = np.copy(scores2nd2)
    # scores2nd3_n = np.copy(scores2nd3)
    scores_n = scores
    scores2nd2_n = scores2nd2
    scores2nd3_n = scores2nd3

    if gold is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if gold[j] != i:
                    scores_n[i][j] += 1

        biarcs2, biarcs3 = xeisner.arcs_to_biarcs(gold)
        for i, k in itertools.product(range(N), range(N)):
            if (i, k) not in biarcs2:
                scores2nd2_n[i, k] += 1.0
            for j in range(N):
                if (i, j, k) not in biarcs3:
                    scores2nd3_n[i, j, k] += 1.0

    return xeisner.eisner2nd_decode(scores_n, scores2nd2_n, scores2nd3_n)


def eisner_native(scores, gold=None):
    from max_sub_tree import xeisner
    N = scores.shape[0] - 1
    scores_n = np.copy(scores)
    if gold is not None:
        for i in range(0, N+1):
            for j in range(1, N+1):
                if gold[j] != i:
                    scores_n[i][j] += 1

    return xeisner.eisner1st_decode(scores_n)


def eisner(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    incomplete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1). 
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1,N+1):
        for s in range(N-k+1):
            t = s+k
            
            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s]==t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t]==s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)
        
    value = complete[0][N][1]
    heads = [-1 for _ in range(N+1)] #-np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in range(1,N+1):
        h = heads[m]
        value_proj += scores[h,m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the 
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return


def chuliu(scores, gold=None, node_status=None):
    node_count = scores.shape[0]

    if gold is not None:
        for i in range(0, node_count):
            for j in range(1, node_count):
                if gold[j] != i:
                    scores[i][j] += 1

    if node_status is None:
        node_status = np.zeros((node_count, ), dtype=np.int8)

    # record max inbound edge
    node_heads = -np.ones((node_count, ), dtype=np.int32)
    for i in range(1, node_count):
        if node_status[i] <= 1:
            for j in range(0, node_count):
                if j == i or node_status[j] >= 2:
                    continue
                if node_heads[i] == -1 or scores[j][i] > scores[node_heads[i]][i]:
                    node_heads[i] = j

    visited = np.zeros((node_count, ), dtype=bool)
    circle = None

    for i in range(0, node_count):
        if visited[i] or node_heads[i] == -1:
            continue

        check_list = []
        j = i
        while j != -1 and not visited[j]:
            check_list.append(j)
            visited[j] = True
            j = node_heads[j]

        # if j == -1:
        #    continue

        try:
            circle_first = check_list.index(j)
        except ValueError:
            continue

        circle = check_list[circle_first:]
        break

    if circle is None:
        return node_heads

    circle_output = sum(scores[i][node_heads[i]] for i in circle)
    last_heads = node_heads.copy()
    for i in range(0, node_count):
        if node_status[i] <= 1:
            node_status[i] = 0
        else:
            node_status[i] = 3

    for i in circle:
        node_status[i] = 2
    node_status[circle[0]] = 1

    new_scores = np.full((node_count, node_count), -1000000, dtype=np.float32)
    real_target = -np.ones((node_count, ), dtype=np.int32)
    real_source = -np.ones((node_count, ), dtype=np.int32)
    for i in range(0, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            if node_status[i] == 0 and node_status[j] == 0:
                new_scores[i][j] = scores[i][j]
            if node_status[i] == 0 and (node_status[j] == 1 or node_status[j] == 2):
                # edge_score = scores[i][j] + circle_output - scores[node_heads[j]][j]
                edge_score = scores[i][j] - scores[node_heads[j]][j]
                if edge_score > new_scores[i][circle[0]]:
                    new_scores[i][circle[0]] = edge_score
                    real_target[i] = j
            if (node_status[i] == 1 or node_status[i] == 2) and node_status[j] == 0:
                edge_score = scores[i][j]
                if edge_score > new_scores[circle[0]][j]:
                    new_scores[circle[0]][j] = edge_score
                    real_source[j] = i

    last_status = node_status.copy()
    node_heads = chuliu(new_scores, gold, node_status)

    x = node_heads[circle[0]]
    y = real_target[x]

    node_heads[circle[0]] = -1
    node_heads[y] = x

    for i in range(0, node_count):
        if (last_status[i] == 1 or last_status[i] == 2) and i != y:
            node_heads[i] = last_heads[i]

    for i in range(0, node_count):
        if node_heads[i] == circle[0] and last_status[i] == 0:
            node_heads[i] = real_source[i]

    return node_heads


def test_chuliu():
    # noinspection PyTypeChecker
    graph = np.array(
            [[min_float, 5, 1, 1],
             [min_float, min_float, 11, 4],
             [min_float, 10, min_float, 5],
             [min_float, 9, 8, min_float]], dtype=np.float32)
    heads = chuliu(graph)
    assert heads == [-1, 0, 1, 2]


decoders = {"eisner": eisner, "eisner_native": eisner_native, "chuliu": chuliu, "eisner2nd": eisner2nd}


if __name__ == '__main__':
    test_chuliu()
