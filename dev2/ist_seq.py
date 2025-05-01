import sys
import itertools
from collections import defaultdict, deque
import subprocess

def write_metis_file(vertices, adj, filename):
    """Write the Bn graph to a file in METIS format."""
    index_map = {v: i+1 for i, v in enumerate(vertices)}  # 1-based indexing
    with open(filename, 'w') as f:
        total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        f.write(f"{len(vertices)} {total_edges}\n")
        for v in vertices:
            neighbors = [str(index_map[nb]) for nb in adj[v]]
            f.write(" ".join(neighbors) + "\n")
    return index_map

def run_metis(input_file, k):
    """Run METIS to partition the graph into k parts."""
    cmd = ["gpmetis", input_file, str(k)]
    subprocess.run(cmd, check=True)
    return f"{input_file}.part.{k}"

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
    index_map = {v: i for i, v in enumerate(vertices)}  # tuple -> index
    reverse_index_map = {i: v for v, i in index_map.items()}  # index -> tuple

    inv, rpos = precompute(vertices, n)
    root = tuple(range(1, n+1))
    trees = {}
    for t in range(1, n):
        parent = {}
        for v in vertices:
            if v == root: continue
            parent[v] = parent1(v, t, n, inv, rpos)
        trees[t] = parent
    return vertices, trees, index_map, reverse_index_map

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

def load_partition_file(filename):
    with open(filename) as f:
        return [int(line.strip()) for line in f]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 ist_seq.py <n> <k_partitions>")
        sys.exit(1)
    
    n = int(sys.argv[1])
    k = int(sys.argv[2])

    # Generate Bn graph and adjacency
    vertices, adj = generate_bn(n)

    # Write METIS format graph
    graph_file = f"Bn_{n}.graph"
    index_map = write_metis_file(vertices, adj, graph_file)

    # Run METIS partitioning
    part_file = run_metis(graph_file, k)

    # Load METIS partition results
    part_labels = load_partition_file(part_file)
    print(f"Partitioned into {k} parts using METIS.")

    # Build ISTs
    vertices, trees, _, _ = build_trees(n)
    print(f"Built {n-1} trees on B_{n} with |V|={len(vertices)}")

    # Optional: Analyze ISTs per partition
    for t, parent in trees.items():
        partition_edges = [v for v in parent if part_labels[index_map[v] - 1] == part_labels[index_map[parent[v]] - 1]]
        print(f"Tree {t}: {len(partition_edges)} intra-partition edges")

    # Check correctness
    if check_tree(n, vertices, trees):
        print(f"SMOKE TEST PASS: Trees are valid for n={n}")
    else:
        print(f"SMOKE TEST FAIL for n={n}")
