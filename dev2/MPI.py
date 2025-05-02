import sys
import subprocess
from mpi4py import MPI
import itertools
from collections import defaultdict, deque
import shutil

def write_metis_file(vertices, adj, filename):
    """Write the Bn graph to a file in METIS format."""
    index_map = {v: i + 1 for i, v in enumerate(vertices)}  # 1-based indexing
    with open(filename, 'w') as f:
        total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        f.write(f"{len(vertices)} {total_edges} 0\n")  # Add format=0
        for v in vertices:
            neighbors = [str(index_map[nb]) for nb in adj[v]]
            f.write(" ".join(neighbors) + "\n")
    return {v: i for i, v in enumerate(vertices)}  # Return 0-based index_map for internal use

def run_metis(input_file, k):
    """Run METIS to partition the graph into k parts."""
    if not shutil.which("gpmetis"):
        raise RuntimeError("METIS not found. Ensure 'gpmetis' is installed and in PATH.")
    subprocess.run(["gpmetis", input_file, str(k)], check=True)
    return f"{input_file}.part.{k}"

def load_partition_file(filename):
    """Load METIS partition labels from file."""
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]

def generate_bn(n):
    """Generate Bn graph vertices and adjacency list."""
    vertices = sorted(itertools.permutations(range(1, n + 1)))
    adj = {v: [] for v in vertices}
    for v in vertices:
        for i in range(n - 1):
            w = list(v)
            w[i], w[i + 1] = w[i + 1], w[i]
            adj[v].append(tuple(w))
    return vertices, adj

def swap(v, x):
    v = list(v)
    idx = v.index(x)
    if idx < len(v) - 1:
        v[idx], v[idx + 1] = v[idx + 1], v[idx]
    return tuple(v)

def parent1(v, t, n, inv, rpos):
    root = tuple(range(1, n + 1))
    vn = v[-1]
    if vn == n:
        if t != n - 1:
            return find_position(v, t, n, inv, rpos)
        else:
            return swap(v, v[-2])
    if vn == n - 1 and v[-2] == n and swap(v, n) != root:
        return swap(v, n) if t == 1 else swap(v, t - 1)
    if vn == t:
        return swap(v, n)
    return swap(v, t)

def find_position(v, t, n, inv, rpos):
    root = tuple(range(1, n + 1))
    if t == 2 and swap(v, t) == root:
        return swap(v, t - 1)
    if v[-2] in {t, n - 1}:
        j = rpos[v]
        return swap(v, j)
    return swap(v, t)

def precompute(vertices, n):
    inv = {}
    rpos = {}
    for v in vertices:
        inv[v] = {v[i]: i + 1 for i in range(n)}  # 1-based positions
        for i in range(n - 1, -1, -1):
            if v[i] != i + 1:
                rpos[v] = i + 1
                break
    return inv, rpos

def construct_local_tree(t, local_vertices, n, inv, rpos):
    root = tuple(range(1, n + 1))
    parent = {}
    for v in local_vertices:
        parent[v] = None if v == root else parent1(v, t, n, inv, rpos)
    return t, parent

def merge_trees(all_trees, n):
    merged = {}
    for part in all_trees:
        for t, tree in part.items():
            merged[t] = tree
    return merged

def save_trees_to_file(merged_ISTs, n, k):
    output_file = f"Bn{n}_ISTs_output.txt"
    with open(output_file, "w") as f:
        f.write(f"Constructed {n - 1} ISTs for n={n} with METIS partitioning (k={k})\n\n")
        for t in sorted(merged_ISTs.keys()):
            f.write(f"Tree t={t} (node → parent):\n")
            tree = merged_ISTs[t]
            for node in sorted(tree.keys()):
                parent = tree[node]
                if parent is None:
                    f.write(f"{node} → ROOT\n")
                else:
                    f.write(f"{node} → {parent}\n")
            f.write("\n")
    print(f"\nAll {n - 1} ISTs have been saved to {output_file}.")

def analyze_edges(tree, partition_labels, index_map):
    intra = inter = 0
    for child, parent in tree.items():
        if parent is None:
            continue

        try:
            if partition_labels[child] == partition_labels[parent]:
                intra += 1
            else:
                inter += 1
        except (IndexError, KeyError) as e:
            print(f"Partition label error: {e} at child={child}, parent={parent}")

    return intra, inter


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpiexec -n <num_procs> python3 MPI.py <n> <k_partitions>")
        comm.Barrier()
        sys.exit(1)

    n = int(sys.argv[1])
    k = int(sys.argv[2])

    if rank == 0:
        vertices, adj = generate_bn(n)
        graph_file = f"Bn_{n}.graph"
        index_map = write_metis_file(vertices, adj, graph_file)
        part_file = run_metis(graph_file, k)
        part_labels = load_partition_file(part_file)
        vertex_label_map = {v: part_labels[i] for i, v in enumerate(vertices)}
    else:
        vertices = None
        index_map = None
        vertex_label_map = None

    # Broadcast full data
    vertices = comm.bcast(vertices, root=0)
    index_map = comm.bcast(index_map, root=0)
    vertex_label_map = comm.bcast(vertex_label_map, root=0)

    # Precompute
    inv, rpos = precompute(vertices, n)

    # Each process works only on its METIS partition
    local_vertices = [v for v in vertices if vertex_label_map[v] == rank]

    # Each process computes parent map for each t=1 to n-1
    local_results = defaultdict(dict)
    for t in range(1, n):
        for v in local_vertices:
            parent = None if v == tuple(range(1, n + 1)) else parent1(v, t, n, inv, rpos)
            local_results[t][v] = parent

    # Gather all partial trees to rank 0
    all_parts = comm.gather(local_results, root=0)

    if rank == 0:
        merged_ISTs = defaultdict(dict)
        for part in all_parts:
            for t, tree in part.items():
                merged_ISTs[t].update(tree)  # Merge partial trees

        save_trees_to_file(merged_ISTs, n, k)

        # Inter/Intra analysis
        print("\nIST Inter/Intra Edge Analysis:")
        for t in range(1, n):
            intra, inter = analyze_edges(merged_ISTs[t], vertex_label_map, index_map)
            print(f"Tree {t}: Intra = {intra}, Inter = {inter}")


