import sys
import subprocess

from mpi4py import MPI
import itertools
from collections import defaultdict
from collections import deque
import pickle

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

def construct_local_tree(t, local_vertices, n, inv, rpos):
    root = tuple(range(1, n+1))
    parent = {}
    for v in local_vertices:
        if v == root:
            parent[v] = None
        else:
            parent[v] = parent1(v, t, n, inv, rpos)
    return t, parent

def merge_trees(all_trees, n):
    """Merge full trees from each process. Each t appears only in one process."""
    merged_ISTs = {}
    for proc_result in all_trees:
        for t, tree in proc_result.items():
            
            merged_ISTs[t] = tree
    return merged_ISTs


def save_trees_to_file(merged_ISTs, n, k):
    """Save the merged trees to an output file."""
    output_file = f"Bn{n}_ISTs_output.txt"
    with open(output_file, "w") as f:
        f.write(f"Constructed {n-1} ISTs for n={n} with METIS partitioning (k={k})\n\n")
        for t in sorted(merged_ISTs.keys()):
            f.write(f"Tree t={t} (node → parent):\n")
            tree = merged_ISTs[t]
            for node in sorted(tree.keys()):
                parent = tree[node]
                if parent is None:
                    f.write(f"{node} → ROOT\n")  # Root node
                else:
                    f.write(f"{node} → {parent}\n")
            f.write("\n")
    print(f"\nAll {n-1} ISTs have been constructed and saved to {output_file}.")

def analyze_edges(tree, partition_labels):
    intra = 0
    inter = 0
    for child, parent in tree.items():
        if parent is None:
            continue  # Skip root node, as it has no parent
        if partition_labels[child] == partition_labels[parent]:
            intra += 1
        else:
            inter += 1
    return intra, inter

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpiexec -n <num_procs> python3 MPI.py <n> <k_partitions>")
        sys.exit(1)

    n = int(sys.argv[1])
    k = int(sys.argv[2])

    vertices, adj = generate_bn(n)

    if rank == 0:
        graph_file = f"Bn_{n}.graph"
        index_map = write_metis_file(vertices, adj, graph_file)
        part_file = run_metis(graph_file, k)
        part_labels = load_partition_file(part_file)
        vertex_label_map = {v: part_labels[i] for i, v in enumerate(vertices)}
    else:
        vertex_label_map = None
        vertices = None

    # Broadcast full data needed for all processes
    vertex_label_map = comm.bcast(vertex_label_map, root=0)
    vertices = comm.bcast(vertices, root=0)

    # Compute helper structures once, globally
    inv, rpos = precompute(vertices, n)

    # Get local vertex set
    local_vertices = [v for v in vertices if vertex_label_map[v] == rank]

    # Assign t-values
    all_t_values = list(range(1, n))  # t = 1 to n-1
    t_per_proc = len(all_t_values) // size
    extra = len(all_t_values) % size
    start = rank * t_per_proc + min(rank, extra)
    end = start + t_per_proc + (1 if rank < extra else 0)
    local_ts = all_t_values[start:end]

    local_results = {}
    local_counts = []

    for t in local_ts:
        tid, tree = construct_local_tree(t, local_vertices, n, inv, rpos)
        intra, inter = analyze_edges(tree, vertex_label_map)
        local_results[tid] = tree
        local_counts.append((tid, intra, inter))

    # Gather at root
    all_trees = comm.gather(local_results, root=0)
    all_counts = comm.gather(local_counts, root=0)

    if rank == 0:
        merged_ISTs = merge_trees(all_trees, n)
        save_trees_to_file(merged_ISTs, n, k)

        print("\nIST Inter/Intra Edge Analysis:")
        for proc_counts in all_counts:
            for tid, intra, inter in proc_counts:
                print(f"Tree {tid}: Intra = {intra}, Inter = {inter}")
