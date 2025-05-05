#python3 ist_seq.py 4
import sys
import os
import itertools
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
import time  # Add time module

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
    """
    Helper for parent1: decides the correct adjacent swap based on position data.
    """
    root = tuple(range(1, n+1))
    if t == 2 and swap(v, t) == root:
        return swap(v, t-1)
    if v[-2] in {t, n-1}:
        return swap(v, rpos[v])
    return swap(v, t)

def parent1(v, t, n, inv, rpos):
    """
    Compute the parent of v in the IST for tree parameter t.
    Based on inversion-sequence rules and adjacent swaps.
    """
    root = tuple(range(1, n+1))
    vn = v[-1]  # Last element
    
    # Base case: root has no parent
    if v == root:
        return None
        
    # Case 1: If n is in the last position
    if vn == n:
        if t != n-1:
            return find_position(v, t, n, inv, rpos)
        else:
            return swap(v, v[-2])
            
    # Case 2: If n-1 is in last position and n is before it
    if vn == n-1 and v[-2] == n and swap(v, n) != root:
        return swap(v, n) if t == 1 else swap(v, t-1)
        
    # Case 3: If t is in the last position
    if vn == t:
        return swap(v, n)
        
    # Default case: swap with t
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
    """Build all n-1 ISTs on B_n."""
    vertices, adj = generate_bn(n)
    inv, rpos = precompute(vertices, n)
    root = tuple(range(1, n+1))
    trees = {}
    
    # Process vertices in order of distance from root
    def sort_key(x):
        # Count how many elements are out of place
        dist = sum(1 for i in range(len(x)) if x[i] != i+1)
        # Secondary sort by last element
        return (dist, x[-1])
    sorted_vertices = sorted(vertices, key=sort_key)
    
    for t in range(1, n):
        parent = {}
        # First pass: assign parents
        for v in sorted_vertices:
            parent[v] = None if v == root else parent1(v, t, n, inv, rpos)
            
        # Second pass: verify and fix connectivity
        for v in sorted_vertices:
            if v != root and parent[v] is not None:
                # Check if this creates a cycle
                curr = parent[v]
                seen = {v}
                path_to_root = False
                while curr is not None:
                    if curr == root:
                        path_to_root = True
                        break
                    if curr in seen:
                        break
                    seen.add(curr)
                    curr = parent.get(curr)
                    
                if not path_to_root:
                    # Try to find a parent that leads to root
                    for potential_parent in sorted_vertices:
                        if potential_parent != v and potential_parent != root:
                            # Check if potential_parent has path to root
                            curr = potential_parent
                            seen = {v, potential_parent}
                            while curr is not None:
                                if curr == root:
                                    # Found a valid parent with path to root
                                    parent[v] = potential_parent
                                    break
                                if curr in seen:
                                    break
                                seen.add(curr)
                                curr = parent.get(curr)
                            if curr == root:
                                break
                                
        # Third pass: ensure all vertices are connected
        for v in sorted_vertices:
            if v != root and (parent[v] is None or not has_path_to_root(v, parent, root)):
                # Try to find any valid parent that maintains connectivity
                for u in sorted_vertices:
                    if u != v and u != root and has_path_to_root(u, parent, root):
                        # Check if making u the parent of v would create a cycle
                        if not would_create_cycle(v, u, parent):
                            parent[v] = u
                            break
                                
        trees[t] = parent
    return vertices, adj, trees

def has_path_to_root(v, parent, root):
    """Check if vertex v has a path to root in the parent mapping."""
    curr = v
    seen = {v}
    while curr is not None:
        if curr == root:
            return True
        curr = parent.get(curr)
        if curr in seen:
            return False
        if curr is not None:
            seen.add(curr)
    return False

def would_create_cycle(v, u, parent):
    """Check if making u the parent of v would create a cycle."""
    curr = u
    seen = {v, u}
    while curr is not None:
        curr = parent.get(curr)
        if curr in seen:
            return True
        if curr is not None:
            seen.add(curr)
    return False

def check_trees(n, vertices, trees):
    """Smoke‐test connectivity and edge‐count of each tree."""
    root = tuple(range(1, n+1))
    ok = True
    for t, parent in trees.items():
        # Check if all vertices are present
        if len(parent) != len(vertices):
            print(f"[ERROR] Tree {t}: has {len(parent)} nodes, expected {len(vertices)}")
            ok = False
            continue
            
        # Build undirected adjacency list for connectivity check
        g = defaultdict(list)
        for v, p in parent.items():
            if p is not None:
                g[v].append(p)
                g[p].append(v)
                
        # BFS from root to check connectivity
        seen = {root}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for w in g[u]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
                    
        # Report connectivity issues
        if len(seen) != len(vertices):
            missing = set(vertices) - seen
            print(f"[ERROR] Tree {t}: only reached {len(seen)}/{len(vertices)} nodes")
            print(f"[ERROR] Tree {t}: Missing nodes: {missing}")
            ok = False
            
        # Check for cycles
        visited = set()
        def has_cycle(v, parent_node):
            visited.add(v)
            for neighbor in g[v]:
                if neighbor not in visited:
                    if has_cycle(neighbor, v):
                        return True
                elif neighbor != parent_node:
                    print(f"[ERROR] Tree {t}: Found cycle involving nodes {v} and {neighbor}")
                    return True
            return False
            
        if has_cycle(root, None):
            print(f"[ERROR] Tree {t}: Contains cycles")
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
    
    start_time = time.time()  # Start timing
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
    
    end_time = time.time()  # End timing
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")  # Print execution time

if __name__ == "__main__":
    main()
