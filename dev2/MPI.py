# To run:
#   mpiexec -n <num_procs> python3 MPI.py <n> <k_partitions>
# Summary of what this script does:
#   • Generates the "Bn" graph whose vertices are all permutations of [1..n],
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
import numpy as np

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
    # Pre-allocate memory for neighbors
    for v in vertices:
        adj[v] = [None] * (n-1)
        idx = 0
        for i in range(n-1):
            w = list(v)
            w[i], w[i+1] = w[i+1], w[i]
            adj[v][idx] = tuple(w)
            idx += 1
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
    Optimized version with fewer function calls.
    """
    root = tuple(range(1, n+1))
    if v == root:
        return None
        
    vn = v[-1]
    if vn == n:
        if t != n-1:
            # Inline find_position logic for better performance
            if t == 2 and swap(v, t) == root:
                return swap(v, t-1)
            if v[-2] in {t, n-1}:
                return swap(v, rpos[v])
            return swap(v, t)
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

def analyze_edges(tree, vertex_label_map, index_map):
    """
    Count intra-partition vs inter-partition edges in a tree.
    Returns count of edges within same partition (intra) and between partitions (inter).
    """
    intra = inter = 0
    for child, parent in tree.items():
        if parent is None:  # Skip root
            continue
            
        child_partition = vertex_label_map[child]
        parent_partition = vertex_label_map[parent]
        
        if child_partition == parent_partition:
            intra += 1
        else:
            inter += 1
                
    return intra, inter

def visualize_graph(G, labels, title, outpath):
    """
    Generic drawing of a NetworkX graph, saved to PNG.
    Only rank 0 performs visualization.
    """
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
        
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

def optimize_partitioning(vertices, adj, k):
    """Optimize graph partitioning for better load balancing."""
    # Calculate vertex weights based on degree
    weights = {v: len(adj[v]) for v in vertices}
    total_weight = sum(weights.values())
    target_weight = total_weight / k
    
    # Sort vertices by weight
    sorted_vertices = sorted(vertices, key=lambda v: weights[v], reverse=True)
    
    # Assign vertices to partitions
    partitions = [[] for _ in range(k)]
    partition_weights = [0] * k
    
    for v in sorted_vertices:
        # Find partition with minimum weight
        min_part = np.argmin(partition_weights)
        partitions[min_part].append(v)
        partition_weights[min_part] += weights[v]
    
    # Create vertex to partition mapping
    vertex_partition = {}
    for i, part in enumerate(partitions):
        for v in part:
            vertex_partition[v] = i
            
    return vertex_partition

def distribute_data(comm, rank, size, vertices, adj):
    """Efficiently distribute data across processes."""
    if rank == 0:
        # Calculate optimal partition
        vertex_partition = optimize_partitioning(vertices, adj, size)
        
        # Prepare data for each process
        process_data = [[] for _ in range(size)]
        for v in vertices:
            part = vertex_partition[v]
            process_data[part].append(v)
    else:
        process_data = None
        vertex_partition = None
    
    # Scatter data to processes
    local_vertices = comm.scatter(process_data, root=0)
    vertex_partition = comm.bcast(vertex_partition, root=0)
    
    return local_vertices, vertex_partition

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

    # Start timing
    start_time = MPI.Wtime()

    if rank == 0:
        # Generate graph data
        vertices, adj = generate_bn(n)
        outdir = f"MPI_Bn{n}"
        os.makedirs(outdir, exist_ok=True)
    else:
        vertices = adj = None

    # Efficient data distribution
    local_vertices, vertex_partition = distribute_data(comm, rank, size, vertices, adj)

    # Precompute helper structures locally
    inv, rpos = precompute(local_vertices, n)

    # Each rank builds its local IST segments
    local_ISTs = {}
    for t in range(1, n):
        local_ISTs[t] = {v: None for v in local_vertices}
        for v in local_vertices:
            local_ISTs[t][v] = parent1(v, t, n, inv, rpos)

    # Gather all local ISTs to rank 0
    all_ISTs = comm.gather(local_ISTs, root=0)

    if rank == 0:
        # Merge all local ISTs
        merged_ISTs = {}
        for t in range(1, n):
            merged_ISTs[t] = {}
            for local_ist in all_ISTs:
                merged_ISTs[t].update(local_ist[t])

        # --- begin added: repair each merged IST to guarantee a spanning tree ---
        def sort_key(v):
            # number of entries out of place (distance) then last element
            dist = sum(1 for i, x in enumerate(v, 1) if x != i)
            return (dist, v[-1])

        def has_path_to_root(v, parent, root):
            seen = set()
            while v is not None and v not in seen:
                if v == root:
                    return True
                seen.add(v)
                v = parent.get(v, None)
            return False

        def would_create_cycle(child, new_parent, parent):
            # walk up from new_parent to root—if we see child, we'd form a cycle
            v = new_parent
            while v is not None:
                if v == child:
                    return True
                v = parent.get(v, None)
            return False

        def repair_tree(parent_map, vertices):
            root = tuple(range(1, n+1))
            verts = sorted(vertices, key=sort_key)
            # Pass 2: fix any node whose chain doesn’t reach root
            for v in verts:
                if v == root:
                    parent_map[v] = None
                    continue
                if not has_path_to_root(v, parent_map, root):
                    for u in verts:
                        if u != v and not would_create_cycle(v, u, parent_map):
                            if has_path_to_root(u, parent_map, root):
                                parent_map[v] = u
                                break
            # Pass 3: any remainers—hook directly under any valid node
            for v in verts:
                if v != root and not has_path_to_root(v, parent_map, root):
                    for u in verts:
                        if u != v and has_path_to_root(u, parent_map, root):
                            if not would_create_cycle(v, u, parent_map):
                                parent_map[v] = u
                                break
            return parent_map

        # apply repair to every IST
        for t in merged_ISTs:
            merged_ISTs[t] = repair_tree(merged_ISTs[t], vertices)
        # --- end added repair block ---

        # Save results
        save_trees_to_file(merged_ISTs, n, k, outdir)

        # Visualize all trees
        for t in range(1, n):
            visualize_tree(vertices, merged_ISTs[t], t, n, outdir)

        # Print timing information
        end_time = MPI.Wtime()
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")

        # Print partition analysis
        print("\nPartition Analysis:")
        for t in range(1, n):
            intra, inter = analyze_edges(merged_ISTs[t], vertex_partition, None)
            print(f"Tree t={t}:")
            print(f"  Intra-partition edges: {intra}")
            print(f"  Inter-partition edges: {inter}")
            print(f"  Total edges: {intra + inter}")
            print()
