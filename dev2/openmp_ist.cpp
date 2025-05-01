// openmp_ist.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// A permutation is just a vector<int>
using Perm = vector<int>;

// Hash function so we can use unordered_map<Perm,int>
struct PermHash {
    size_t operator()(Perm const& p) const noexcept {
        size_t h = p.size();
        for (auto &x : p) {
            // a simple rolling hash
            h = h * 31 + x;
        }
        return h;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <n> <k_partitions>\n";
        return 1;
    }
    int n = stoi(argv[1]);
    int k = stoi(argv[2]);

    //
    // 1) Generate all permutations of [1..n], sorted lexicographically
    //
    vector<Perm> vertices;
    Perm p(n);
    iota(p.begin(), p.end(), 1);
    function<void(int)> gen = [&](int idx) {
        if (idx == n) {
            vertices.push_back(p);
            return;
        }
        for (int i = idx; i < n; ++i) {
            swap(p[idx], p[i]);
            gen(idx+1);
            swap(p[idx], p[i]);
        }
    };
    gen(0);
    sort(vertices.begin(), vertices.end());

    int V = vertices.size();  // = n!

    // Build a map from permutation -> its index (0..V-1)
    unordered_map<Perm,int,PermHash> index_map(V);
    for (int i = 0; i < V; ++i) {
        index_map[vertices[i]] = i;
    }

    //
    // 2) Precompute rpos[i] = rightmost position where vertices[i][pos] != pos+1
    //
    vector<int> rpos(V, 0);
    for (int i = 0; i < V; ++i) {
        auto &v = vertices[i];
        for (int pos = n-1; pos >= 0; --pos) {
            if (v[pos] != pos+1) {
                rpos[i] = pos+1;  // 1-based
                break;
            }
        }
    }

    //
    // 3) Write the graph in METIS format to "Bn_<n>.graph"
    //
    string graph_file = "Bn_" + to_string(n) + ".graph";
    {
        ofstream mf(graph_file);
        // each vertex has (n-1) neighbors, but edges are undirected:
        int m = V * (n - 1) / 2;
        mf << V << " " << m << " 0\n";
        for (int i = 0; i < V; ++i) {
            auto &v = vertices[i];
            // list neighbors by swapping adjacent elements
            for (int j = 0; j < n - 1; ++j) {
                Perm w = v;
                swap(w[j], w[j+1]);
                int nbr_idx = index_map[w] + 1; // METIS uses 1-based
                mf << nbr_idx << (j < n-2 ? " " : "");
            }
            mf << "\n";
        }
    }

    //
    // 4) Run gpmetis to partition into k parts
    //
    {
        string cmd = "gpmetis " + graph_file + " " + to_string(k);
        if (system(cmd.c_str()) != 0) {
            cerr << "Error: failed to run gpmetis.\n";
            return 1;
        }
    }

    //
    // 5) Read back the partition file
    //
    vector<int> part_label(V);
    {
        string part_file = graph_file + ".part." + to_string(k);
        ifstream pf(part_file);
        for (int i = 0; i < V; ++i) {
            pf >> part_label[i];
        }
    }

    //
    // 6) Helpers for IST construction
    //
    // Swap-adjacent-x in permutation v
    auto swap_adj = [&](Perm const &v, int x) {
        Perm w = v;
        // find x in w
        auto it = find(w.begin(), w.end(), x);
        int idx = int(it - w.begin());
        if (idx < n - 1) swap(w[idx], w[idx+1]);
        return w;
    };
    // find_position logic
    auto find_position = [&](Perm const &v, int t) {
        Perm root(n);
        iota(root.begin(), root.end(), 1);
        if (t == 2 && swap_adj(v, t) == root) {
            return swap_adj(v, t-1);
        }
        // if v[n-2] in {t, n-1}
        int vid = index_map[v];
        if (v[n-2] == t || v[n-2] == n-1) {
            int j = rpos[vid];
            return swap_adj(v, j);
        }
        return swap_adj(v, t);
    };
    // parent1 logic (mirrors your Python version)
    auto parent1 = [&](Perm const &v, int t) {
        Perm root(n);
        iota(root.begin(), root.end(), 1);
        int vid = index_map[v];
        int vn = v[n-1];
        if (vn == n) {
            if (t != n - 1) return find_position(v, t);
            else return swap_adj(v, v[n-2]);
        }
        if (vn == n-1 && v[n-2] == n && swap_adj(v, n) != root) {
            if (t == 1) return swap_adj(v, n);
            else return swap_adj(v, t-1);
        }
        if (vn == t) {
            return swap_adj(v, n);
        }
        return swap_adj(v, t);
    };

    //
    // 7) Build all ISTs in parallel with OpenMP
    //
    // IST[t][i] = index of parent of vertex i in tree t (t=1..n-1)
    vector<vector<int>> IST(n, vector<int>(V, -1));

    // Precompute the ROOT permutation
    Perm root(n);
    iota(root.begin(), root.end(), 1);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int t = 1; t < n; ++t) {
        for (int i = 0; i < V; ++i) {
            if (vertices[i] == root) {
                IST[t][i] = -1;  // root has no parent
            } else {
                Perm par = parent1(vertices[i], t);
                IST[t][i] = index_map[par];
            }
        }
    }

    //
    // 8) Save ISTs to file
    //
    {
        ofstream of("Bn" + to_string(n) + "_ISTs_output.txt");
        of << "Constructed " << (n - 1)
           << " ISTs for n=" << n
           << " with METIS partitioning (k=" << k << ")\n\n";

        for (int t = 1; t < n; ++t) {
            of << "Tree t=" << t << " (node → parent):\n";
            for (int i = 0; i < V; ++i) {
                // print the vertex
                of << "(";
                for (int j = 0; j < n; ++j) {
                    of << vertices[i][j]
                       << (j + 1 < n ? "," : "");
                }
                of << ") → ";
                int pi = IST[t][i];
                if (pi < 0) {
                    of << "ROOT\n";
                } else {
                    of << "(";
                    for (int j = 0; j < n; ++j) {
                        of << vertices[pi][j]
                           << (j + 1 < n ? "," : "");
                    }
                    of << ")\n";
                }
            }
            of << "\n";
        }
    }

    //
    // 9) Print inter-/intra-partition edge counts
    //
    cout << "\nIST Inter/Intra Edge Analysis:\n";
    for (int t = 1; t < n; ++t) {
        int intra = 0, inter = 0;
        for (int i = 0; i < V; ++i) {
            int pi = IST[t][i];
            if (pi < 0) continue;
            if (part_label[i] == part_label[pi]) ++intra;
            else ++inter;
        }
        cout << "Tree " << t
             << ": Intra = " << intra
             << ", Inter = " << inter << "\n";
    }

    return 0;
}

