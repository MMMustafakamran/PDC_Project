#python3 ist_seq.py 4
import sys
import os
import itertools
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

def generate_bn(n):
    """Generate vertices and adjacency list of B_n."""
    vertices = sorted(itertools.permutations(range(1, n+1)))
    adj = {v: [] for v in vertices}
    for v in vertices:
        for i in range(n-1):
            w = list(v)
            w[i], w[i+1] = w[i+1], w[i]
            adj[v].append(tuple(w))
    return vertices, adj

def swap(v, x):
    v = list(v)
    idx = v.index(x)
    if idx < len(v)-1:
        v[idx], v[idx+1] = v[idx+1], v[idx]
    return tuple(v)

def find_position(v, t, n, inv, rpos):
    root = tuple(range(1, n+1))
    # Implements “FindPosition” from the paper
    if t == 2 and swap(v, t) == root:
        return swap(v, t-1)
    if v[-2] in {t, n-1}:
        j = rpos[v]
        return swap(v, j)
    return swap(v, t)

def parent1(v, t, n, inv, rpos):
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

def precompute(vertices, n):
    """Compute inverse permutations and rightmost out‐of‐place pos."""
    inv = {}
    rpos = {}
    for v in vertices:
        inv[v] = {v[i]: i+1 for i in range(n)}
        # rightmost position i where v[i] != i+1
        for i in range(n-1, -1, -1):
            if v[i] != i+1:
                rpos[v] = i+1
                break
    return inv, rpos

def build_trees(n):
    vertices, adj = generate_bn(n)
    inv, rpos = precompute(vertices, n)
    root = tuple(range(1, n+1))
    trees = {}
    for t in range(1, n):
        parent = {}
        for v in vertices:
            parent[v] = None if v == root else parent1(v, t, n, inv, rpos)
        trees[t] = parent
    return vertices, adj, trees

def check_trees(n, vertices, trees):
    """Smoke‐test connectivity and edge‐count of each tree."""
    root = tuple(range(1, n+1))
    ok = True
    for t, parent in trees.items():
        if len(parent) != len(vertices):
            print(f"[ERROR] Tree {t}: has {len(parent)} nodes, expected {len(vertices)}")
            ok = False
        # build undirected adjacency
        g = defaultdict(list)
        for v, p in parent.items():
            if p is not None:
                g[v].append(p)
                g[p].append(v)
        # BFS
        seen = {root}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for w in g[u]:
                if w not in seen:
                    seen.add(w)
                    dq.append(w)
        if len(seen) != len(vertices):
            print(f"[ERROR] Tree {t}: only reached {len(seen)}/{len(vertices)} nodes")
            ok = False
    return ok

def visualize_bn(vertices, adj, n, outdir):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for v in vertices:
        for w in adj[v]:
            if v < w:
                G.add_edge(v, w)
    labels = {v: ''.join(map(str, v)) for v in vertices}
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=50, linewidths=0.2)
    nx.draw_networkx_edges(G, pos, width=0.2)
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    plt.title(f"Full B_{n} Graph ({len(vertices)} nodes)")
    plt.axis('off')
    fn = os.path.join(outdir, f"Bn{n}_full_graph.png")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Saved full‐graph to {fn}")

def visualize_tree(vertices, parent, t, n, outdir):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for v, p in parent.items():
        if p is not None:
            G.add_edge(v, p)
    labels = {v: ''.join(map(str, v)) for v in vertices}
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=50, linewidths=0.2)
    nx.draw_networkx_edges(G, pos, width=0.2)
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    plt.title(f"IST Tree t={t} on B_{n}")
    plt.axis('off')
    fn = os.path.join(outdir, f"Bn{n}_ist_t{t}.png")
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Saved IST‐tree t={t} to {fn}")

def save_parents_file(trees, n, outdir):
    fn = os.path.join(outdir, f"Bn{n}_IST_parents.txt")
    with open(fn, "w") as f:
        f.write(f"Constructed {n-1} ISTs on B_{n}\n\n")
        for t in sorted(trees):
            f.write(f"Tree t={t} (node -> parent):\n")
            for v in sorted(trees[t]):
                p = trees[t][v]
                if p is None:
                    f.write(f"{v} -> ROOT\n")
                else:
                    f.write(f"{v} -> {p}\n")
            f.write("\n")
    print(f"Saved parent mapping to {fn}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 ist_seq.py <n>")
        sys.exit(1)
    n = int(sys.argv[1])
    vertices, adj, trees = build_trees(n)
    print(f"Built {n-1} ISTs on B_{n} with |V| = {len(vertices)} vertices")

    if check_trees(n, vertices, trees):
        print("SMOKE TEST PASS: All trees are valid")
    else:
        print("SMOKE TEST FAIL: Issues detected")

    outdir = f"Seq_Bn{n}"
    os.makedirs(outdir, exist_ok=True)

    visualize_bn(vertices, adj, n, outdir)
    for t, parent in trees.items():
        visualize_tree(vertices, parent, t, n, outdir)

    save_parents_file(trees, n, outdir)

if __name__ == "__main__":
    main()
