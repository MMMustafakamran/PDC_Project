/*
    Compilation and Execution Example:


    mpic++ -std=c++11 -O3 -fopenmp openmp_c++.cpp -o openmp_ist
    export OMP_NUM_THREADS=4
    mpiexec -n 4 ./openmp_ist 4 2

    # Run with MPI (e.g., 2 MPI processes, n=6, k=2 partitions - PNGs skipped)
    mpiexec -n 2 ./openmp_ist 6 2
*/
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mkdir
#include <cerrno>      // For errno
#include <cstring>     // For strerror

// --- Platform specific directory creation ---
#ifdef _WIN32
#include <direct.h> // For _mkdir
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755) // POSIX
#endif
// -----------------------------------------


// Type aliases
using Perm    = std::vector<int>;
using AdjList = std::vector<std::vector<int>>;

// Directory for all output files
const std::string OUTPUT_DIR = "openMp";

// ——————— Utility routines ———————

std::vector<Perm> gen_vertices(int n) {
    Perm p(n);
    std::iota(p.begin(), p.end(), 1);
    std::vector<Perm> V;
    V.reserve(std::tgamma(n + 1)); // Pre-allocate roughly n! space
    do { V.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    return V;
}

// Helper function to convert permutation to string representation
std::string perm_to_string(const Perm& p) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < p.size(); ++i) {
        oss << p[i] << (i < p.size() - 1 ? " " : "");
    }
    oss << "]";
    return oss.str();
}

AdjList gen_adj(const std::vector<Perm>& V) {
    int N = V.size(), n = V[0].size();
    AdjList adj(N);
    // Optimization: Precompute neighbors in parallel if N is large
    // #pragma omp parallel for // Be careful with race conditions if modifying shared state directly
    for (int i = 0; i < N; ++i) {
        adj[i].reserve(n - 1); // Reserve space for neighbors
        for (int j = 0; j < n - 1; ++j) {
            Perm w = V[i];
            std::swap(w[j], w[j + 1]);
            // Use lower_bound for potentially faster search in sorted V
            auto it = std::lower_bound(V.begin(), V.end(), w);
            // Ensure the found element is actually the one we are looking for
            if (it != V.end() && *it == w) {
                 adj[i].push_back(int(it - V.begin()));
            } else {
                 // This should ideally not happen if V contains all permutations
                 std::cerr << "Warning: Neighbor permutation not found for " << perm_to_string(V[i]) << " swapping indices " << j << " and " << j+1 << std::endl;
            }
        }
        // Sort neighbors for consistency (optional, but can help METIS)
        std::sort(adj[i].begin(), adj[i].end());
    }
    return adj;
}



// Partitions the graph using METIS, placing files in OUTPUT_DIR
std::vector<int> partition_metis(const AdjList& adj, int k) {
    int N = adj.size();
    if (N == 0) {
        std::cerr << "Error: Cannot partition an empty graph." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string metis_input_file = OUTPUT_DIR + "/bn.metis";
    std::string metis_part_file = metis_input_file + ".part." + std::to_string(k);

    std::cout << "Rank 0: Writing graph to METIS format file: " << metis_input_file << std::endl;
    std::ofstream fout(metis_input_file);
     if (!fout) {
        std::cerr << "Rank 0: ERROR - Cannot open METIS input file for writing: " << metis_input_file << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long edge_count = 0; // Use long long for potentially large edge counts
    for (const auto& nbrs : adj) {
        edge_count += nbrs.size();
    }
    edge_count /= 2; // METIS counts edges once

    // METIS format: N M [fmt], fmt=0 means no weights
    fout << N << " " << edge_count << " 0\n";
    for (const auto& nbrs : adj) {
        for (int j : nbrs) {
            fout << (j + 1) << " "; // METIS uses 1-based indexing
        }
        fout << "\n";
    }
    fout.close();
    std::cout << "Rank 0: Finished writing METIS file. Nodes: " << N << ", Edges: " << edge_count << std::endl;

    // Construct the gpmetis command
    std::string cmd = "gpmetis \"" + metis_input_file + "\" " + std::to_string(k);
    std::cout << "Rank 0: Running METIS partitioner: " << cmd << std::endl;

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Rank 0: ERROR - gpmetis execution failed! Exit code: " << ret << ". Check METIS installation and file permissions." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
     std::cout << "Rank 0: gpmetis finished." << std::endl;


    std::vector<int> part(N);
    std::cout << "Rank 0: Reading METIS partition file: " << metis_part_file << std::endl;
    std::ifstream pin(metis_part_file);
     if (!pin) {
        std::cerr << "Rank 0: ERROR - Cannot open METIS partition file: " << metis_part_file << ". Check if gpmetis ran successfully." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N; ++i) {
        if (!(pin >> part[i])) {
             std::cerr << "Rank 0: ERROR - Failed to read partition data for node " << i << " from " << metis_part_file << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    pin.close();
     std::cout << "Rank 0: Successfully read partition data." << std::endl;
    return part;
}

void precompute(const std::vector<Perm>& V,
                std::vector<std::vector<int>>& inv,
                std::vector<int>& rpos) {
    int N = V.size();
    if (N == 0) return; // Handle empty V
    int n = V[0].size();
    inv.assign(N, std::vector<int>(n + 1));
    rpos.assign(N, 0);
    #pragma omp parallel for // Parallelize precomputation
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[i][ V[i][j] ] = j; // Use 0-based index internally for simplicity
        }
        for (int j = n - 1; j >= 0; --j) {
            if (V[i][j] != j + 1) { // Check if element is in its "sorted" position
                rpos[i] = j; // Store 0-based index of rightmost non-fixed element
                break;
            }
        }
         // If all elements are fixed (identity permutation), rpos[i] remains 0 (implicit or explicit)
         // Note: Original code set rpos to j+1 (1-based). Adjusted to 0-based index.
         // Ensure parent1_cpp logic aligns with 0-based or 1-based rpos.
         // Sticking to original logic: use 1-based index for rpos.
         for (int j = n - 1; j >= 0; --j) {
             if (V[i][j] != j + 1) {
                 rpos[i] = j + 1; // Use 1-based index as in original code
                 break;
             }
         }
         // Also update inv to use 1-based index position for consistency with original parent logic
         for (int j = 0; j < n; ++j) {
             inv[i][ V[i][j] ] = j + 1;
         }
    }
}


// Finds the index of the permutation resulting from swapping element x with its right neighbor
int swap_perm(int vi, int x, const std::vector<Perm>& V, int n, const std::vector<std::vector<int>>& inv) {
    // Optimization: Use precomputed inverse to find index of x quickly
    int idx = inv[vi][x] - 1; // Get 0-based index from 1-based inv

    if (idx < 0 || idx >= n) {
        std::cerr << "Error in swap_perm: Element " << x << " not found or index out of bounds for permutation " << perm_to_string(V[vi]) << std::endl;
        // Handle error appropriately, maybe return -1 or throw exception
         return -1; // Indicate error
    }

    Perm w = V[vi]; // Copy the permutation
    if (idx + 1 < n) { // Check if swap is valid (not the last element)
        std::swap(w[idx], w[idx + 1]);
    } else {
        // If x is the last element, the "swap" results in the same permutation
        return vi;
    }

    // Find the index of the resulting permutation w in the sorted list V
    auto pos = std::lower_bound(V.begin(), V.end(), w);
     if (pos != V.end() && *pos == w) {
        return int(pos - V.begin());
     } else {
         std::cerr << "Error in swap_perm: Resulting permutation " << perm_to_string(w) << " not found in V." << std::endl;
         return -1; // Indicate error
     }
}

// The core parent function (adapted slightly for clarity and using precomputed inv)
int parent1_cpp(int vi, int t, int n,
                const std::vector<Perm>& V,
                const std::vector<std::vector<int>>& inv,
                const std::vector<int>& rpos) {
    const int root = 0; // Index of the identity permutation [1, 2, ..., n]
    if (vi == root) return -1; // Root has no parent

    const Perm& v = V[vi];
    int vn = v.back(); // Value of the last element (at index n-1)

    // Helper lambda for swapping, includes error checking
    auto safe_swap_perm = [&](int current_vi, int element_to_swap) {
        int result_idx = swap_perm(current_vi, element_to_swap, V, n, inv);
        if (result_idx < 0) {
             std::cerr << "Error: swap_perm failed during parent calculation for v=" << perm_to_string(V[current_vi]) << ", t=" << t << ", element=" << element_to_swap << std::endl;
            // Decide how to handle: return error code, specific value, or abort?
             return -2; // Use a distinct error code
        }
        return result_idx;
    };

    if (vn == n) { // Case 1: n is the last element
        if (t != n - 1) {
            int rpos_val = rpos[vi]; // This is 1-based index from original logic
             int rpos_idx = rpos_val > 0 ? rpos_val - 1 : -1; // Convert to 0-based index if valid
             int val_at_rpos = (rpos_idx >= 0 && rpos_idx < n) ? v[rpos_idx] : -1; // Value at rightmost non-fixed position

            if (t == 2 && safe_swap_perm(vi, t) == root) {
                return safe_swap_perm(vi, t - 1); // Swap with 1
            }
             // Check value at index n-2 (second to last)
             int val_n_minus_2 = (n >= 2) ? v[n - 2] : -1;
             if (val_n_minus_2 == t || val_n_minus_2 == n - 1) {
                 if (val_at_rpos == -1) { // Should not happen if not root
                      std::cerr << "Error: Invalid rpos value for non-root permutation " << perm_to_string(v) << std::endl;
                      return -2;
                 }
                return safe_swap_perm(vi, val_at_rpos); // Swap with element at rpos
             }
            return safe_swap_perm(vi, t); // General case: swap with t
        } else { // t == n - 1
             int val_n_minus_2 = (n >= 2) ? v[n - 2] : -1;
             if (val_n_minus_2 == -1) { // Handle n=1 case? (Though typically n>=2)
                 std::cerr << "Warning: parent1_cpp called with n < 2?" << std::endl;
                 return -1; // Or appropriate handling
             }
            return safe_swap_perm(vi, val_n_minus_2); // Swap with element at index n-2
        }
    }

    // Case 2: n is not the last element
     int swap_n_result = safe_swap_perm(vi, n);
     if (swap_n_result == -2) return -2; // Propagate error

    if (vn == n - 1 && n >= 2 && v[n - 2] == n && swap_n_result != root) {
        return (t == 1 ? swap_n_result : safe_swap_perm(vi, t - 1));
    }
    if (vn == t) {
        return swap_n_result; // Swap with n
    }
    return safe_swap_perm(vi, t); // Default: swap with t
}


// Checks if following parents from v eventually leads back to v (detects cycle involving np)
bool would_cycle(int child, int np, const std::vector<int>& parent) {
    int N = parent.size();
    int current = np;
    // Limit iterations to prevent infinite loops in case of unexpected graph structure
    for(int i = 0; i < N && current >= 0; ++i) {
        if (current == child) return true; // Found the child in the ancestor path of np
        current = parent[current];
    }
    return false; // Did not find child in ancestor path
}

// Checks if following parents from v leads to the root (-1 parent)
bool has_path_to_root(int v, const std::vector<int>& parent, int /*root_ignored*/) {
     int N = parent.size();
     int current = v;
     // Limit iterations to prevent infinite loops
     for (int i = 0; i < N && current >= 0; ++i) {
         current = parent[current];
     }
     return (current < 0); // Path exists if we reached a node with parent -1
}


// Repair function to ensure the parent array forms a valid spanning tree rooted at index 0
void repair_tree(std::vector<int>& par, int n, const std::vector<Perm>& V, const AdjList& adj, const std::vector<std::vector<int>>& inv) {
    int N = par.size();
    if (N == 0) return;
    int root = 0; // Assuming root is always index 0 (identity permutation)

    std::vector<int> q; q.reserve(N); // Queue for BFS/DFS-like traversal
    std::vector<bool> visited(N, false);
    std::vector<int> potential_parents; potential_parents.reserve(n - 1);

    // Pass 1: Identify nodes disconnected from the main tree component containing the root
    q.push_back(root);
    visited[root] = true;
    par[root] = -1; // Ensure root's parent is -1

    int head = 0;
    while(head < q.size()){
        int u = q[head++];
        // Iterate through all nodes to find children of u in the current 'par' array
        for(int v = 0; v < N; ++v){
            if(!visited[v] && par[v] == u){
                // Check for cycles before adding to prevent infinite loops if input 'par' has cycles
                 bool cycle = false;
                 int ancestor = par[u];
                 for(int k=0; k<N && ancestor >=0; ++k){
                     if(ancestor == v) { cycle = true; break; }
                     ancestor = par[ancestor];
                 }

                 if(!cycle){
                    visited[v] = true;
                    q.push_back(v);
                 } else {
                     // Found a cycle involving v -> u -> ... -> v
                     // Mark v as needing repair (disconnect it for now)
                     // par[v] = -2; // Or some indicator, handled later
                     std::cerr << "Repair Warning: Detected cycle involving node " << v << " (" << perm_to_string(V[v]) << ") during initial traversal. It will be re-parented." << std::endl;
                     // Don't mark as visited yet, let the next phase handle it.
                 }
            }
        }
    }

    // Pass 2: Re-parent nodes that were not visited (disconnected or part of cycles broken above)
    bool changed = true;
    int repair_passes = 0;
    const int MAX_REPAIR_PASSES = N; // Safety break

    while(changed && repair_passes < MAX_REPAIR_PASSES){
        changed = false;
        repair_passes++;
         std::cout << "Rank 0: Starting repair pass " << repair_passes << std::endl; // DEBUG
        for (int v = 0; v < N; ++v) {
            if (!visited[v]) { // If node v is not connected to the root component
                 // Find a potential parent u among its neighbors in the original graph (adj)
                 // such that u IS visited (connected to root) and adding v->u doesn't create a cycle.
                 potential_parents.clear();
                 for(int neighbor_idx : adj[v]){
                     potential_parents.push_back(neighbor_idx);
                 }
                 // Optional: Shuffle potential parents to avoid bias?
                 // std::random_shuffle(potential_parents.begin(), potential_parents.end());

                int best_parent = -1;
                for (int u : potential_parents) {
                    if (visited[u] && !would_cycle(v, u, par)) {
                        best_parent = u;
                        break; // Found a valid parent
                    }
                }

                if (best_parent != -1) {
                    par[v] = best_parent;
                    visited[v] = true; // Now it's connected
                    changed = true;
                     // std::cout << "Rank 0: Repaired node " << v << " (" << perm_to_string(V[v]) << ") -> parent " << best_parent << " (" << perm_to_string(V[best_parent]) << ")" << std::endl; // Verbose Debug
                } else {
                     // This case should be rare if the original graph is connected.
                     // It might indicate an issue or a severely disconnected component.
                      std::cerr << "Rank 0: Repair Warning - Node " << v << " (" << perm_to_string(V[v]) << ") could not find a valid parent in repair pass " << repair_passes << ". Still disconnected." << std::endl;
                }
            }
        }
        // After potentially connecting more nodes, update the visited set fully
        // This is a bit inefficient, ideally only update newly connected nodes,
        // but a full BFS ensures correctness if cycles were complex.
         if(changed){
              q.clear();
              head = 0;
              std::fill(visited.begin(), visited.end(), false); // Reset visited for re-traversal
              q.push_back(root);
              visited[root] = true;
               while(head < q.size()){
                   int u = q[head++];
                   for(int v = 0; v < N; ++v){
                       if(!visited[v] && par[v] == u){
                           visited[v] = true;
                           q.push_back(v);
                       }
                   }
               }
         }

    } // End while changed

     if(repair_passes == MAX_REPAIR_PASSES && changed){
          std::cerr << "Rank 0: Warning - Repair process reached max passes, potential issues remain." << std::endl;
     }

     // Final check: Ensure all nodes are visited (connected)
     int disconnected_count = 0;
      for(int v=0; v<N; ++v){
          if(!visited[v]) disconnected_count++;
      }
      if(disconnected_count > 0){
           std::cerr << "Rank 0: ERROR - After repair, " << disconnected_count << " nodes remain disconnected from the root!" << std::endl;
           // Optional: List disconnected nodes
            for(int v=0; v<N; ++v){
                if(!visited[v]) std::cerr << "  - Node " << v << " (" << perm_to_string(V[v]) << ")" << std::endl;
            }
           // Consider aborting if complete connectivity is essential
           // MPI_Abort(MPI_COMM_WORLD, 1);
      } else {
           std::cout << "Rank 0: Repair process completed. All nodes appear connected." << std::endl;
      }

}


// ——————— GraphViz visualization Utilities ———————

// Checks if the 'dot' command is available in the system path
bool check_dot_available() {
    // Redirect stderr to stdout, then to /dev/null (or NUL on Windows) to suppress output
    #ifdef _WIN32
        int ret = std::system("dot -V > NUL 2>&1");
    #else
        int ret = std::system("dot -V > /dev/null 2>&1");
    #endif
    return (ret == 0);
}

// Writes DOT file and optionally generates PNG for an IST
void write_ist_dot_and_png(const std::vector<int>& parent,
                           int t, int N,
                           const std::string& dir, // Use OUTPUT_DIR passed in
                           bool generate_png,     // Flag controls PNG generation
                           const std::vector<Perm>& V) {
    // 1) Write DOT file (always)
    std::ostringstream dot_filename_stream;
    dot_filename_stream << dir << "/ist_t" << t << ".dot";
    std::string dot_filename = dot_filename_stream.str();

    std::ofstream dot_out(dot_filename);
     if (!dot_out) {
        std::cerr << "Rank 0: Error - Could not open file " << dot_filename << " for writing." << std::endl;
        return;
    }

    dot_out << "digraph IST_t" << t << " {\n"
            << "  rankdir=TB;\n"
            << "  label=\"IST for t=" << t << " (n=" << (V.empty() ? 0 : V[0].size()) << ")\";\n"
            << "  node [shape=box, fontname=\"Courier\"];\n"; // Use box shape and monospace font

    // Create node definitions with permutation labels
    for (int i = 0; i < N; ++i) {
        std::string perm_label = perm_to_string(V[i]);
        // Add ROOT indicator in label if it's the root
        if (parent[i] < 0) {
            dot_out << "  \"" << i << "\" [label=\"" << perm_label << "\\n(ROOT)\"];\n"; // Use index as node ID
        } else {
            dot_out << "  \"" << i << "\" [label=\"" << perm_label << "\"];\n"; // Use index as node ID
        }
    }

    // Create edges using node indices
    for (int i = 0; i < N; ++i) {
        if (parent[i] >= 0) { // If node i has a parent
            // Ensure parent index is valid
            if (parent[i] >= N) {
                 std::cerr << "Rank 0: Error in write_ist_dot_and_png - Invalid parent index " << parent[i] << " for node " << i << std::endl;
                 continue; // Skip this invalid edge
            }
            dot_out << "  \"" << parent[i] << "\" -> \"" << i << "\";\n"; // Edge from parent index to child index
        }
    }

    dot_out << "}\n";
    dot_out.close();
    std::cout << "Rank 0: Written IST t=" << t << " definition to " << dot_filename << std::endl;

    // 2) Render PNG if requested and possible
    if (generate_png) {
        std::ostringstream png_filename_stream;
        png_filename_stream << dir << "/ist_t" << t << ".png";
        std::string png_filename = png_filename_stream.str();

        std::ostringstream cmd_stream;
        // Enclose filenames in quotes for robustness with spaces/special chars
        cmd_stream << "dot -Tpng \"" << dot_filename << "\" -o \"" << png_filename << "\"";
        std::string cmd = cmd_stream.str();

         std::cout << "Rank 0: Attempting to generate PNG: " << cmd << std::endl;
        int ret = std::system(cmd.c_str());
         if (ret != 0) {
             std::cerr << "Rank 0: Warning - Graphviz PNG generation failed for IST t=" << t << ". Command exit code: " << ret << std::endl;
         } else {
             std::cout << "Rank 0: Written IST t=" << t << " visualization to " << png_filename << std::endl;
         }
    }
}

// ——————— Main Program ———————
#include <iostream>
int main(int argc, char** argv) {
   
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto global_start_time = std::chrono::high_resolution_clock::now();

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: mpiexec -n <num_processes> " << argv[0] << " <n> <k>\n"
                      << "  <n>: Size of permutations (e.g., 3 for S_3)\n"
                      << "  <k>: Number of partitions for METIS (should be <= num_processes)\n"
                      << "Example: mpiexec -n 4 " << argv[0] << " 4 4\n";
        }
        MPI_Finalize();
        return 1;
    }

    int n = 0;
    int k = 0;
    try {
        n = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
    } catch (const std::exception& e) {
         if (rank == 0) std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
         return 1;
    }


    if (n <= 0) {
        if (rank == 0) std::cerr << "Error: n must be a positive integer." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
     if (k <= 0) {
        if (rank == 0) std::cerr << "Error: k (number of partitions) must be a positive integer." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
     // It's often good if k matches the number of MPI processes, but not strictly required by METIS
     if (k > size && rank == 0) {
         std::cout << "Warning: Number of partitions k=" << k << " is greater than the number of MPI processes=" << size << ". METIS will run, but process mapping might be uneven." << std::endl;
     }
     if (k == 1 && size > 1 && rank == 0) {
          std::cout << "Info: Running with k=1 partition. MPI parallelism benefit might be limited after partitioning." << std::endl;
     }


    if (rank == 0) {
        std::cout << "============================================\n"
                  << " Running IST Generation for B_n (n=" << n << ")\n"
                  << " MPI Processes: " << size << "\n"
                  << " OpenMP Threads per process: " << omp_get_max_threads() << "\n"
                  << " METIS Partitions (k): " << k << "\n"
                  << " Output Directory: " << OUTPUT_DIR << "\n"
                  << "============================================" << std::endl;
    }


    // --- Check for `dot` and Create Output Directory (Rank 0) ---
    bool can_dot = false;
    if (rank == 0) {
        can_dot = check_dot_available();
        if (!can_dot) {
            std::cout << "Rank 0: Warning - GraphViz `dot` command not found in PATH. Skipping ALL PNG generation." << std::endl;
        } else {
             std::cout << "Rank 0: Found `dot` command. PNG generation enabled (for n <= 5)." << std::endl;
        }

        // Create output directory
        int mkdir_ret = MKDIR(OUTPUT_DIR.c_str());
        if (mkdir_ret != 0 && errno != EEXIST) {
            std::cerr << "Rank 0: Error creating output directory '" << OUTPUT_DIR << "': " << strerror(errno) << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            if (errno == EEXIST) {
                std::cout << "Rank 0: Output directory '" << OUTPUT_DIR << "' already exists." << std::endl;
            } else {
                std::cout << "Rank 0: Created output directory '" << OUTPUT_DIR << "'." << std::endl;
            }
            errno = 0; // Reset errno after checking EEXIST
        }
    }
    // Broadcast dot availability status to all ranks
    MPI_Bcast(&can_dot, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    // --- Generate Vertices, Adjacency List, and Partition (Rank 0) ---
    std::vector<Perm> V;
    AdjList adj;
    std::vector<int> part; // Partition assignment for each vertex
    int N = 0;             // Total number of vertices (n!)

    auto stage_start_time = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::cout << "\nRank 0: Stage 1: Generating Vertices and Adjacency List..." << std::endl;
        V   = gen_vertices(n);
        N   = V.size(); // Calculate N = n!
        std::cout << "Rank 0: Generated " << N << " vertices (permutations)." << std::endl;
        adj = gen_adj(V);
        std::cout << "Rank 0: Generated adjacency list." << std::endl;

        


        std::cout << "\nRank 0: Stage 2: Partitioning Graph using METIS (k=" << k << ")..." << std::endl;
        part = partition_metis(adj, k); // Generates bn.metis and bn.metis.part.k in OUTPUT_DIR
        std::cout << "Rank 0: Graph partitioning complete." << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stage_start_time);
        std::cout << "Rank 0: Stages 1 & 2 duration: " << duration.count() << " ms" << std::endl;
    }

    // --- Broadcast N and Partition Data ---
    stage_start_time = std::chrono::high_resolution_clock::now();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Resize partition vector on non-zero ranks before receiving data
        part.resize(N);
    }
    MPI_Bcast(part.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\nRank 0: Broadcasted N and partition data to all ranks." << std::endl;


    // --- Scatter Vertex Indices Based on Partition ---
    // Calculate send counts and displacements for Scatterv
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<int> vertex_indices(N); // Global list of vertex indices [0, 1, ..., N-1]

    if (rank == 0) {
        std::iota(vertex_indices.begin(), vertex_indices.end(), 0); // Fill with 0, 1, ..., N-1
        // Count how many vertices belong to each partition (process rank)
        for (int i = 0; i < N; ++i) {
            int target_rank = part[i];
             if (target_rank >= 0 && target_rank < size) {
                 sendcounts[target_rank]++;
             } else {
                  std::cerr << "Rank 0: Warning - METIS partition index " << part[i] << " for vertex " << i << " is out of MPI rank bounds [0, " << size - 1 << "]. Assigning to rank 0." << std::endl;
                  // Assign problematic nodes to rank 0 (or handle differently)
                  part[i] = 0;
                  sendcounts[0]++;
             }

        }
        // Calculate displacements (starting index for each rank's data in the global buffer)
        displs[0] = 0;
        for (int r = 1; r < size; ++r) {
            displs[r] = displs[r - 1] + sendcounts[r - 1];
        }

        // --- Reorder vertex_indices according to partition for contiguous scattering ---
        // This step is crucial for Scatterv to work correctly if `part` isn't already sorted by rank
        std::vector<int> reordered_indices(N);
        std::vector<int> current_pos = displs; // Use displs to track where to put next index for each rank
        for(int i=0; i<N; ++i){
            int target_rank = part[i];
            reordered_indices[current_pos[target_rank]] = vertex_indices[i]; // Place original index `i`
            current_pos[target_rank]++;
        }
         vertex_indices = std::move(reordered_indices); // Use the reordered list
         //-----------------------------------------------------------------------------

         std::cout << "Rank 0: Calculated scatter counts and displacements." << std::endl;
         // Debug print counts/displs
         // for(int r=0; r<size; ++r) std::cout << "  Rank " << r << ": count=" << sendcounts[r] << ", displ=" << displs[r] << std::endl;
    }

    // Broadcast sendcounts and displacements needed by Scatterv on all ranks
    MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for local vertex indices on each rank
    std::vector<int> local_vertex_indices(sendcounts[rank]);

    // Scatter the reordered vertex indices
    MPI_Scatterv(rank == 0 ? vertex_indices.data() : nullptr, // Send buffer only on rank 0
                 sendcounts.data(),
                 displs.data(),
                 MPI_INT,
                 local_vertex_indices.data(), // Receive buffer on each rank
                 sendcounts[rank],           // Size of receive buffer
                 MPI_INT,
                 0,                         // Root rank
                 MPI_COMM_WORLD);

    std::cout << "Rank " << rank << ": Received " << local_vertex_indices.size() << " vertex indices." << std::endl;


    // --- Precomputation (All Ranks Need V, inv, rpos) ---
     if (rank != 0) {
          // Generate vertices locally if not rank 0
          V = gen_vertices(n);
          if(V.size() != N) { // Sanity check
              std::cerr << "Rank " << rank << ": ERROR - Generated |V|=" << V.size() << " inconsistent with N=" << N << " from Rank 0." << std::endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
          }
     }
    std::vector<std::vector<int>> inv; // Inverse permutation table
    std::vector<int> rpos;             // Rightmost non-fixed position

    std::cout << "Rank " << rank << ": Stage 3: Precomputing inv/rpos tables..." << std::endl;
    precompute(V, inv, rpos);
    std::cout << "Rank " << rank << ": Precomputation finished." << std::endl;
    auto duration_scatter_precompute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stage_start_time);
    std::cout << "Rank " << rank << ": Scatter + Precompute duration: " << duration_scatter_precompute.count() << " ms" << std::endl;


    // --- Build Local ISTs in Parallel using OpenMP ---
    stage_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Rank " << rank << ": Stage 4: Building local ISTs (t=1 to " << n-1 << ") using " << omp_get_max_threads() << " OpenMP threads..." << std::endl;
    std::vector<std::vector<int>> local_parent_arrays(n); // Index t -> parent array for local vertices

    #pragma omp parallel for schedule(dynamic)
    for (int t = 1; t < n; ++t) {
        int thread_id = omp_get_thread_num(); // Optional: for debugging thread activity
        local_parent_arrays[t].resize(local_vertex_indices.size());
        for (int i = 0; i < (int)local_vertex_indices.size(); ++i) {
            int global_vertex_index = local_vertex_indices[i];
            local_parent_arrays[t][i] = parent1_cpp(global_vertex_index, t, n, V, inv, rpos);
             // Error check result from parent1_cpp
             if(local_parent_arrays[t][i] == -2) {
                  #pragma omp critical
                  {
                      std::cerr << "Rank " << rank << ", Thread " << thread_id << ": ERROR in parent1_cpp calculation for global_vertex=" << global_vertex_index << ", t=" << t << ". Aborting." << std::endl;
                  }
                  MPI_Abort(MPI_COMM_WORLD, 1); // Abort all processes on critical error
             }
        }
         // Optional: Log completion per thread/t
         // #pragma omp critical
         // {
         //     std::cout << "Rank " << rank << ", Thread " << thread_id << ": Finished local IST for t=" << t << std::endl;
         // }
    }
    std::cout << "Rank " << rank << ": Finished building all local ISTs." << std::endl;
    auto duration_local_build = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stage_start_time);
     std::cout << "Rank " << rank << ": Local IST build duration: " << duration_local_build.count() << " ms" << std::endl;


    // --- Gather Results to Rank 0 ---
    stage_start_time = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "\nRank 0: Stage 5: Gathering local IST results..." << std::endl;
    std::vector<std::vector<int>> global_merged_IST(n); // Index t -> complete parent array on Rank 0

    // Need recvcounts and recvdispls for Gatherv (same as sendcounts/displs but from rank 0's perspective)
    std::vector<int> recvcounts = sendcounts; // Same counts as scatter
    std::vector<int> recvdispls = displs;   // Same displacements as scatter


    for (int t = 1; t < n; ++t) {
        // Rank 0 allocates space for the full merged array for this 't'
        if (rank == 0) {
            global_merged_IST[t].resize(N);
        }

        // Gather the local parent arrays into the correct positions in the global array on rank 0
        MPI_Gatherv(local_parent_arrays[t].data(),      // Send buffer (local results for tree t)
                    local_parent_arrays[t].size(),      // Send count for this rank
                    MPI_INT,
                    rank == 0 ? global_merged_IST[t].data() : nullptr, // Receive buffer (only on rank 0)
                    recvcounts.data(),                  // Array of receive counts from each rank
                    recvdispls.data(),                  // Array of displacements for each rank's data
                    MPI_INT,
                    0,                                  // Root rank
                    MPI_COMM_WORLD);
         MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks finish Gatherv before rank 0 proceeds for this t
         if(rank == 0) std::cout << "Rank 0: Gathered results for IST t=" << t << std::endl;
    }

    if (rank == 0) {
         std::cout << "Rank 0: Finished gathering all IST results." << std::endl;
         auto duration_gather = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stage_start_time);
         std::cout << "Rank 0: Gather duration: " << duration_gather.count() << " ms" << std::endl;
    }


    // --- Repair, Save, and Visualize (Rank 0 only) ---
    if (rank == 0) {
        stage_start_time = std::chrono::high_resolution_clock::now();
        std::cout << "\nRank 0: Stage 6: Repairing and Writing ISTs (t=1 to " << n-1 << ")..." << std::endl;

         // We need adjacency list `adj` again for the repair function
         // If adj is large, consider broadcasting it earlier or regenerating if memory is tight.
         // Assuming adj is available from Stage 1 on rank 0.
         if(adj.empty() && N > 0) {
              std::cerr << "Rank 0: ERROR - Adjacency list `adj` is missing for repair step. Aborting." << std::endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
         }

        // --- Reconstruct the final parent array based on original indices ---
        // The gathered array global_merged_IST[t] is currently ordered according
        // to the `vertex_indices` scattered earlier. We need to map it back
        // so that final_IST[t][i] is the parent of vertex `i`.
        std::vector<std::vector<int>> final_IST(n, std::vector<int>(N));
        std::vector<int> global_to_original_map(N); // Map scattered position back to original vertex index
        for(int r=0; r<size; ++r){
            for(int i=0; i<recvcounts[r]; ++i){
                 global_to_original_map[recvdispls[r] + i] = vertex_indices[recvdispls[r] + i]; // vertex_indices holds the original index
            }
        }

        for (int t = 1; t < n; ++t) {
            std::vector<int> temp_final(N);
             for(int i=0; i<N; ++i){
                 int original_vertex_index = global_to_original_map[i];
                 temp_final[original_vertex_index] = global_merged_IST[t][i];
             }
             final_IST[t] = std::move(temp_final);
        }
        std::cout << "Rank 0: Reordered gathered ISTs to match original vertex indices." << std::endl;
        // --- End Reordering ---


        int root_node_index = 0; // Assuming identity permutation [1,2,...,n] is index 0

        for (int t = 1; t < n; ++t) {
            std::cout << "\nRank 0: Processing IST t=" << t << "..." << std::endl;

            std::cout << "Rank 0: Repairing tree t=" << t << "..." << std::endl;
            repair_tree(final_IST[t], n, V, adj, inv); // Use the reordered final_IST
            std::cout << "Rank 0: Finished repairing tree t=" << t << "." << std::endl;


            // --- Save Text Representation ---
            std::string txt_filename = OUTPUT_DIR + "/ist_t" + std::to_string(t) + ".txt";
            std::ofstream fout_txt(txt_filename);
             if (!fout_txt) {
                 std::cerr << "Rank 0: Error - Could not open file " << txt_filename << " for writing." << std::endl;
             } else {
                 fout_txt << "# Independent Spanning Tree (IST) for t = " << t << " in B_" << n << "\n";
                 fout_txt << "# Format: vertex_index vertex_permutation -> parent_index parent_permutation\n";
                 for (int i = 0; i < N; ++i) {
                     fout_txt << i << " " << perm_to_string(V[i]) << " -> ";
                     int parent_idx = final_IST[t][i]; // Use the repaired final_IST
                     if (parent_idx < 0) {
                         fout_txt << "ROOT\n";
                     } else if (parent_idx >= N) {
                          fout_txt << "INVALID_PARENT_INDEX(" << parent_idx << ")\n"; // Should not happen after repair
                     } else {
                         fout_txt << parent_idx << " " << perm_to_string(V[parent_idx]) << "\n";
                     }
                 }
                 fout_txt.close();
                 std::cout << "Rank 0: Written IST t=" << t << " text representation to " << txt_filename << std::endl;
            }


            // --- Write DOT and conditionally PNG ---
            bool generate_ist_png = (n <= 5 && can_dot);
            // Pass the final, repaired, and correctly indexed parent array
            write_ist_dot_and_png(final_IST[t], t, N, OUTPUT_DIR, generate_ist_png, V);

            if (!generate_ist_png) {
                 if (n > 5) {
                      std::cout << "Rank 0: Skipped IST t=" << t << " PNG generation (n > 5)." << std::endl;
                 } else if (!can_dot){
                      // Message about dot availability already shown
                 }
            }
        } // End loop over t

        auto duration_repair_write = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stage_start_time);
        std::cout << "\nRank 0: Stage 6 (Repair & Write) duration: " << duration_repair_write.count() << " ms" << std::endl;

    } // End if (rank == 0) for final stage

    MPI_Barrier(MPI_COMM_WORLD); // Ensure rank 0 finishes before others exit

    if (rank == 0) {
        auto global_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end_time - global_start_time);
        std::cout << "\n============================================" << std::endl;
        std::cout << " Total Execution Time: " << total_duration.count() << " ms" << std::endl;
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}