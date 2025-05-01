import sys
import itertools
from collections import defaultdict, deque

def generate_bn(n):
    """Generate vertices and adjacency list of B_n."""
    vertices = list(itertools.permutations(range(1, n+1)))
    adj = {v: [] for v in vertices}
    for v in vertices:
        for i in range(n-1):
            w = list(v)
            w[i], w[i+1] = w[i+1], w[i]
            adj[v].append(tuple(w))
    return vertices, adj

def swap(v, x):
    """Swap symbol x in v with its successor in the permutation."""
    v = list(v)
    idx = v.index(x)
    if idx < len(v)-1:
        v[idx], v[idx+1] = v[idx+1], v[idx]
    return tuple(v)

def parent1(v, t, n, inv, rpos):
    """Compute the Parent1 of vertex v in tree T^n_t"""
    root = tuple(range(1, n+1))
    vn = v[-1]
    # Rule (1)
    if vn == n:
        if t != n-1:
            return find_position(v, t, n, inv, rpos)
        else:
            return swap(v, v[-2])
    # Rule (2)
    if vn == n-1 and v[-2] == n and swap(v, n) != root:
        if t == 1:
            return swap(v, n)
        else:
            return swap(v, t-1)
    # Rule (5)
    if vn == t:
        return swap(v, n)
    # Rule (6)
    return swap(v, t)

def find_position(v, t, n, inv, rpos):
    """Function FindPosition(v) per the paper."""
    root = tuple(range(1, n+1))
    if t == 2 and swap(v, t) == root:
        return swap(v, t-1)
    if v[-2] in {t, n-1}:
        j = rpos[v]
        return swap(v, j)
    return swap(v, t)

def precompute(vertices, n):
    """Compute inverse permutations and rpos (rightmost out-of-place)."""
    inv = {}
    rpos = {}
    for v in vertices:
        inv[v] = {v[i]: i+1 for i in range(n)}  # symbol->1-based pos
        for i in range(n-1, -1, -1):
            if v[i] != i+1:
                rpos[v] = i+1
                break
    return inv, rpos

def build_trees(n):
    vertices, _ = generate_bn(n)
    inv, rpos = precompute(vertices, n)
    root = tuple(range(1, n+1))
    trees = {}
    for t in range(1, n):
        parent = {}
        for v in vertices:
            if v == root: continue
            parent[v] = parent1(v, t, n, inv, rpos)
        trees[t] = parent
    return vertices, trees


def check_tree(n, vertices, trees):
    root = tuple(range(1, n+1))
    ok = True
    for t, parent in trees.items():
        # Check edge count
        if len(parent) != len(vertices)-1:
            print(f"Tree {t}: wrong edge count {len(parent)} vs {len(vertices)-1}")
            ok = False
        # Check connectivity
        g = defaultdict(list)
        for v, p in parent.items():
            g[v].append(p)
            g[p].append(v)
        visited = set([root])
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for w in g[u]:
                if w not in visited:
                    visited.add(w)
                    dq.append(w)
        if len(visited) != len(vertices):
            print(f"Tree {t}: not connected ({len(visited)}/{len(vertices)})")
            ok = False
    return ok

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 ist_seq.py <n>")
        sys.exit(1)
    n = int(sys.argv[1])
    vertices, trees = build_trees(n)
    print(f"Built {n-1} trees on B_{n} with |V|={len(vertices)}")
    if check_tree(n, vertices, trees):
        print(f"SMOKE TEST PASS: Trees are valid for n={n}")
    else:
        print(f"SMOKE TEST FAIL for n={n}")