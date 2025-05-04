
# To run:
#   mpiexec -n <num_procs> python3 MPI.py <n> <k_partitions>
# Summary of what this script does:
#   • Generates the “Bn” graph whose vertices are all permutations of [1..n],
#     with edges between permutations that differ by one adjacent swap.
#   • Writes the Bn graph in METIS format and uses gpmetis to partition it into k parts.
#   • Broadcasts the graph and partition labels to all MPI processes.
#   • Each MPI rank computes, for each t=1..n-1, a local piece of the
#     Inversion-Sequence Tree (IST) parent map using adjacent-swap rules.
#   • Gathers all partial ISTs on rank 0, merges them, and:
#       – Saves each full IST to a text file.
#       – Visualizes the full Bn graph and each IST as PNGs.
#       – Reports intra- vs. inter-partition edge counts for each IST.

import sys
import subprocess
import shutil
import os
import itertools
from collections import defaultdict
from mpi4py import MPI
import networkx as nx
import matplotlib.pyplot as plt

def write_metis_file(vertices, adj, filename):
    """
    Write the Bn graph to a file in METIS format (1-based indexing).
    Returns a 0-based index_map for internal use.
    """
    index_map = {v: i + 1 for i, v in enumerate(vertices)}  # 1-based for METIS
    with open(filename, 'w') as f:
        total_edges = sum(len(neigh) for neigh in adj.values()) // 2
        f.write(f"{len(vertices)} {total_edges} 0\n")  # format=0
        for v in vertices:
            neighbors = [str(index_map[nb]) for nb in adj[v]]
            f.write(" ".join(neighbors) + "\n")
    # Return 0-based index_map for use in partition analysis
    return {v: i for i, v in enumerate(vertices)}

def run_metis(input_file, k):
    """
    Invoke gpmetis to partition the graph into k parts.
    """
    if not shutil.which("gpmetis"):
        raise RuntimeError("METIS not found. Install gpmetis and ensure it's in PATH.")
    subprocess.run(["gpmetis", input_file, str(k)], check=True)
    return f"{input_file}.part.{k}"

def load_partition_file(filename):
    """
    Read METIS output and return a list of integer labels per node.
    """
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]

def generate_bn(n):
    """
    Build the Bn graph:
      – vertices = all permutations of [1..n]
      – adj[v] = list of neighbors via single adjacent swap
    """
    vertices = sorted(itertools.permutations(range(1, n+1)))
    adj = {v: [] for v in vertices}
    for v in vertices:
        for i in range(n-1):
            w = list(v)
            w[i], w[i+1] = w[i+1], w[i]
            adj[v].append(tuple(w))
    return vertices, adj

def swap(v, x):
    """
    Swap element x with its successor in the tuple v (if possible).
    """
    v = list(v)
    idx = v.index(x)
    if idx < len(v)-1:
        v[idx], v[idx+1] = v[idx+1], v[idx]
    return tuple(v)

def parent1(v, t, n, inv, rpos):
    """
    Compute the parent of v in the IST for tree parameter t.
    Based on inversion-sequence rules and adjacent swaps.
    """
    root = tuple(range(1, n+1))
    vn = v[-1]
    if vn == n:
        if t != n-1:
            return find_position(v, t, n, inv, rpos)
        else:
            return swap(v, v[-2])
    if vn == n-1 and v[-2] == n and swap(v, n) != root:
        return swap(v, n) if t == 1 else swap(v, t-1)
    if vn == t:
        return swap(v, n)
    return swap(v, t)

def find_position(v, t, n, inv, rpos):
    """
    Helper for parent1: decides the correct adjacent swap based on position data.
    """
    root = tuple(range(1, n+1))
    if t == 2 and swap(v, t) == root:
        return swap(v, t-1)
    if v[-2] in {t, n-1}:
        j = rpos[v]
        return swap(v, j)
    return swap(v, t)

def precompute(vertices, n):
    """
    Build:
      – inv[v]: element→position mapping (1-based) for each permutation v
      – rpos[v]: rightmost index i where v[i] != i+1
    """
    inv = {}
    rpos = {}
    for v in vertices:
        inv[v] = {v[i]: i+1 for i in range(n)}
        for i in range(n-1, -1, -1):
            if v[i] != i+1:
                rpos[v] = i+1
                break
    return inv, rpos

def analyze_edges(tree, partition_labels, index_map):
    """
    Count intra-partition vs inter-partition edges in a tree.
    """
    intra = inter = 0
    for child, parent in tree.items():
        if parent is None:
            continue
        if partition_labels[index_map[child]] == partition_labels[index_map[parent]]:
            intra += 1
        else:
            inter += 1
    return intra, inter

def visualize_graph(G, labels, title, outpath):
    """
    Generic drawing of a NetworkX graph, saved to PNG.
    """
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=300, linewidths=0.5)
    nx.draw_networkx_edges(G, pos, width=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def visualize_bn(vertices, adj, n, outdir):
    """
    Build and save the full Bn graph visualization.
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for v in vertices:
        for w in adj[v]:
            if v < w:
                G.add_edge(v, w)
    labels = {v: ''.join(map(str, v)) for v in vertices}
    outpath = os.path.join(outdir, f"bn{n}_graph.png")
    visualize_graph(G, labels, f"Full B_{n} Graph", outpath)

def visualize_tree(vertices, parent, t, n, outdir):
    """
    Build and save the IST for a particular t.
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for v, p in parent.items():
        if p is not None:
            G.add_edge(v, p)
    labels = {v: ''.join(map(str, v)) for v in vertices}
    outpath = os.path.join(outdir, f"bn{n}_ist_t{t}.png")
    visualize_graph(G, labels, f"IST t={t}", outpath)

def save_trees_to_file(merged_ISTs, n, k, outdir):
    """
    Write all merged IST parent maps to a text file.
    """
    outfile = os.path.join(outdir, f"bn{n}_ist_parents_k{k}.txt")
    with open(outfile, 'w') as f:
        f.write(f"ISTs for n={n}, k={k} partitions\n\n")
        for t in sorted(merged_ISTs):
            f.write(f"Tree t={t}:\n")
            for node, par in sorted(merged_ISTs[t].items()):
                f.write(f"  {node} → {par if par else 'ROOT'}\n")
            f.write("\n")

# -----------------------------------------------------------------------------
# Main MPI workflow
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Expect exactly two arguments: n and k
    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpiexec -n <num_procs> python3 MPI.py <n> <k_partitions>")
        comm.Barrier()
        sys.exit(1)

    n = int(sys.argv[1])
    k = int(sys.argv[2])

    if rank == 0:
        # — Rank 0 builds the graph and runs METIS —
        outdir = f"MPI_Bn{n}"
        os.makedirs(outdir, exist_ok=True)

        vertices, adj = generate_bn(n)
        visualize_bn(vertices, adj, n, outdir)

        graph_file = os.path.join(outdir, f"bn{n}_metis_graph.txt")
        index_map = write_metis_file(vertices, adj, graph_file)

        part_file = run_metis(graph_file, k)
        part_labels = load_partition_file(part_file)
        vertex_label_map = {v: part_labels[i] for i, v in enumerate(vertices)}
    else:
        vertices = index_map = vertex_label_map = None

    # Broadcast graph data to all ranks
    vertices         = comm.bcast(vertices, root=0)
    index_map        = comm.bcast(index_map, root=0)
    vertex_label_map = comm.bcast(vertex_label_map, root=0)

    # Precompute helper structures
    inv, rpos = precompute(vertices, n)

    # Filter to the subset assigned to this rank
    local_vertices = [v for v in vertices if vertex_label_map[v] == rank]

    # Each rank builds its local IST segments
    local_results = defaultdict(dict)
    for t in range(1, n):
        for v in local_vertices:
            parent = None if v == tuple(range(1, n+1)) else parent1(v, t, n, inv, rpos)
            local_results[t][v] = parent

    # Gather all partial ISTs at rank 0
    all_parts = comm.gather(local_results, root=0)

    if rank == 0:
        # Merge and save/visualize all ISTs
        merged = defaultdict(dict)
        for part in all_parts:
            for t, tree in part.items():
                merged[t].update(tree)

        save_trees_to_file(merged, n, k, outdir)
        for t, tree in merged.items():
            visualize_tree(vertices, tree, t, n, outdir)

        # Report inter/intra-partition stats
        print("\nIST Inter/Intra Edge Analysis:")
        for t in range(1, n):
            intra, inter = analyze_edges(merged[t], vertex_label_map, index_map)
            print(f"  Tree {t}: Intra = {intra}, Inter = {inter}")
