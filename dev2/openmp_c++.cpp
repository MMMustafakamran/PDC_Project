/*
   mpic++ -O3 -fopenmp openmp_c++.cpp -o openmp_ist

export OMP_NUM_THREADS=4
mpiexec -n 4 ./openmp_ist 6 4
*/
// File: openmp_ist.cpp
// Compile with:
//   mpic++ -O3 -fopenmp openmp_ist.cpp -o openmp_ist
// Run with:
//   export OMP_NUM_THREADS=4
//   mpiexec -n 4 ./openmp_ist 6 4

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>

// Type aliases
using Perm    = std::vector<int>;
using AdjList = std::vector<std::vector<int>>;

// ——————— Utility routines ———————

std::vector<Perm> gen_vertices(int n) {
    Perm p(n);
    std::iota(p.begin(), p.end(), 1);
    std::vector<Perm> V;
    do { V.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    return V;
}

AdjList gen_adj(const std::vector<Perm>& V) {
    int N = V.size(), n = V[0].size();
    AdjList adj(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            Perm w = V[i];
            std::swap(w[j], w[j + 1]);
            auto it = std::lower_bound(V.begin(), V.end(), w);
            adj[i].push_back(int(it - V.begin()));
        }
    }
    return adj;
}

std::vector<int> partition_metis(const AdjList& adj, int k, int /*n*/) {
    int N = adj.size();
    std::ofstream fout("bn.metis");
    int M = 0;
    for (auto &nbrs : adj) M += nbrs.size();
    M /= 2;
    fout << N << " " << M << " 0\n";
    for (auto &nbrs : adj) {
        for (int j : nbrs) fout << (j + 1) << " ";
        fout << "\n";
    }
    fout.close();

    // ** FIXED: build the command string, then pass c_str() to system() **
    std::string cmd = "gpmetis bn.metis " + std::to_string(k);
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "ERROR: gpmetis failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> part(N);
    std::ifstream pin("bn.metis.part." + std::to_string(k));
    for (int i = 0; i < N; ++i) pin >> part[i];
    return part;
}

void precompute(const std::vector<Perm>& V,
                std::vector<std::vector<int>>& inv,
                std::vector<int>& rpos) {
    int N = V.size(), n = V[0].size();
    inv.assign(N, std::vector<int>(n + 1));
    rpos.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j)
            inv[i][ V[i][j] ] = j + 1;
        for (int j = n - 1; j >= 0; --j) {
            if (V[i][j] != j + 1) {
                rpos[i] = j + 1;
                break;
            }
        }
    }
}

int swap_perm(int vi, int x, const std::vector<Perm>& V) {
    Perm w = V[vi];
    auto it = std::find(w.begin(), w.end(), x);
    int idx = it - w.begin();
    if (idx + 1 < (int)w.size()) std::swap(w[idx], w[idx + 1]);
    auto pos = std::lower_bound(V.begin(), V.end(), w);
    return int(pos - V.begin());
}

int parent1_cpp(int vi, int t, int n,
                const std::vector<Perm>& V,
                const std::vector<std::vector<int>>& inv,
                const std::vector<int>& rpos) {
    int root = 0;
    if (vi == root) return -1;
    const Perm& v = V[vi];
    int vn = v.back();
    if (vn == n) {
        if (t != n - 1) {
            if (t == 2 && swap_perm(vi, t, V) == root)
                return swap_perm(vi, t - 1, V);
            if (v[n - 2] == t || v[n - 2] == n - 1)
                return swap_perm(vi, rpos[vi], V);
            return swap_perm(vi, t, V);
        } else {
            return swap_perm(vi, v[n - 2], V);
        }
    }
    if (vn == n - 1 && v[n - 2] == n && swap_perm(vi, n, V) != root) {
        return (t == 1 ? swap_perm(vi, n, V) : swap_perm(vi, t - 1, V));
    }
    if (vn == t) {
        return swap_perm(vi, n, V);
    }
    return swap_perm(vi, t, V);
}

bool has_path_to_root(int v, const std::vector<int>& parent, int root) {
    std::vector<bool> seen(parent.size(), false);
    while (v >= 0 && !seen[v]) {
        if (v == root) return true;
        seen[v] = true;
        v = parent[v];
    }
    return false;
}
bool would_cycle(int child, int np, const std::vector<int>& parent) {
    int v = np;
    while (v >= 0) {
        if (v == child) return true;
        v = parent[v];
    }
    return false;
}

void repair_tree(std::vector<int>& par, int /*n*/, int root) {
    int N = par.size();
    std::vector<int> verts(N);
    std::iota(verts.begin(), verts.end(), 0);
    for (int pass = 0; pass < 2; ++pass) {
        for (int v : verts) {
            if (v == root) { par[v] = -1; continue; }
            if (!has_path_to_root(v, par, root)) {
                for (int u : verts) {
                    if (u != v && !would_cycle(v, u, par)
                        && has_path_to_root(u, par, root)) {
                        par[v] = u;
                        break;
                    }
                }
            }
        }
    }
}

// ——————— GraphViz visualization with guard ———————

bool check_dot_available() {
    int ret = std::system("dot -V > /dev/null 2>&1");
    return (ret == 0);
}

void write_dot_and_png(const std::vector<int>& parent,
                       int t, int N,
                       const std::string& dir,
                       bool can_dot) {
    // 1) Write DOT (always)
    std::ostringstream dotname;
    dotname << dir << "/ist_t" << t << ".dot";
    std::ofstream dot(dotname.str());
    dot << "digraph IST_t" << t << " {\n"
        << "  rankdir=TB;\n";
    for (int i = 0; i < N; ++i) {
        if (parent[i] < 0) {
            dot << "  " << i << " [label=\"" << i << "\\nROOT\"];\n";
        } else {
            dot << "  " << parent[i] << " -> " << i << ";\n";
        }
    }
    dot << "}\n";
    dot.close();

    // 2) Render PNG if available
    if (can_dot) {
        std::ostringstream cmd;
        cmd << "dot -Tpng " << dotname.str()
            << " -o " << dir << "/ist_t" << t << ".png";
        std::system(cmd.str().c_str());
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc != 3) {
        if (rank == 0)
            std::cerr<<"Usage: mpiexec -n <np> ./openmp_ist <n> <k>\n";
        MPI_Finalize();
        return 1;
    }
    int n = std::stoi(argv[1]);
    int k = std::stoi(argv[2]);

    // Check for `dot` on rank 0
    bool can_dot = false;
    if (rank == 0) {
        can_dot = check_dot_available();
        if (!can_dot) {
            std::cerr<<"Warning: GraphViz `dot` not found; skipping PNG generation.\n";
        }
    }
    MPI_Bcast(&can_dot, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    // Generate & partition
    std::vector<Perm> V;
    AdjList adj;
    std::vector<int> part;
    int N = 0;
    if (rank == 0) {
        V   = gen_vertices(n);
        adj = gen_adj(V);
        N   = V.size();
        mkdir(("OpenMp_Bn" + std::to_string(n)).c_str(), 0755);
        part = partition_metis(adj, k, n);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) part.resize(N);
    MPI_Bcast(part.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter vertices
    std::vector<int> sendc(size,0), displ(size,0);
    for (int i = 0; i < N; ++i) sendc[part[i]]++;
    for (int r = 1; r < size; ++r) displ[r] = displ[r-1] + sendc[r-1];
    std::vector<int> idx_all(N), local_idx(sendc[rank]);
    std::iota(idx_all.begin(), idx_all.end(), 0);
    MPI_Scatterv(idx_all.data(), sendc.data(), displ.data(), MPI_INT,
                 local_idx.data(), sendc[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    // Precompute
    if (rank != 0) {
        V   = gen_vertices(n);
        adj = gen_adj(V);
    }
    std::vector<std::vector<int>> inv;
    std::vector<int> rpos;
    precompute(V, inv, rpos);

    // Build local ISTs
    std::vector<std::vector<int>> local_IST(n);
    #pragma omp parallel for schedule(dynamic)
    for (int t = 1; t < n; ++t) {
        local_IST[t].resize(local_idx.size());
        for (int i = 0; i < (int)local_idx.size(); ++i) {
            local_IST[t][i] = parent1_cpp(
                local_idx[i], t, n, V, inv, rpos
            );
        }
    }

    // Gather
    std::vector<std::vector<int>> merged_IST(n);
    for (int t = 1; t < n; ++t) {
        std::vector<int> flat = local_IST[t];
        int loc_sz = flat.size();
        std::vector<int> all_sz(size), all_disp(size);
        MPI_Gather(&loc_sz, 1, MPI_INT, all_sz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            all_disp[0] = 0;
            for (int r = 1; r < size; ++r)
                all_disp[r] = all_disp[r-1] + all_sz[r-1];
            merged_IST[t].resize(N);
        }
        MPI_Gatherv(flat.data(), loc_sz, MPI_INT,
                    rank==0 ? merged_IST[t].data() : nullptr,
                    all_sz.data(), all_disp.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    // Repair, save, visualize
    if (rank == 0) {
        std::string outdir = "OpenMp_Bn" + std::to_string(n);
        int root = 0;
        for (int t = 1; t < n; ++t) {
            repair_tree(merged_IST[t], n, root);

            // Save text
            std::ofstream fout(outdir + "/ist_t" + std::to_string(t) + ".txt");
            fout << "node -> parent\n";
            for (int i = 0; i < N; ++i) {
                fout << i << " -> "
                     << (merged_IST[t][i] < 0 ? "ROOT"
                                              : std::to_string(merged_IST[t][i]))
                     << "\n";
            }
            fout.close();

            // Write DOT + optionally PNG
            write_dot_and_png(merged_IST[t], t, N, outdir, can_dot);
            std::cout << "Written IST t=" << t
                      << (can_dot ? " + PNG\n" : " (PNG skipped)\n");
        }
    }

    MPI_Finalize();
    return 0;
}
