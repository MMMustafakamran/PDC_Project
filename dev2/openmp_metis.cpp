// openmp_metis.cpp
/*
compile: mpicxx -O3 -march=native -std=c++17 -fopenmp openmp_metis.cpp -lmetis -o openmp_metis
run: mpirun -np 4 ./openmp_metis 5 4

Summary:
- Generates all permutations of size n and builds a graph where vertices represent permutations.
- Partitions the graph using METIS for distributed processing.
- Computes parent permutations in parallel using OpenMP.
- Gathers results and writes them to a file in "openMP_Bn<n>/Bn<n>_ISTs_Seq_Parents.txt".
*/
#include <mpi.h>
#include <omp.h>
#include <metis.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// A permutation is a vector<int> of size n, values 1..n
using Perm = std::vector<int>;

// Hash for Perm so we can use it as a key in unordered_map
namespace std {
  template<> struct hash<Perm> {
    size_t operator()(Perm const& v) const noexcept {
      size_t h = 0;
      for (auto &x : v) {
        h ^= std::hash<int>()(x) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
      }
      return h;
    }
  };
}

// Generate all n! permutations of [1..n]
static std::vector<Perm> generate_all_perms(int n) {
  Perm p(n);
  for (int i = 0; i < n; ++i) p[i] = i+1;
  std::vector<Perm> all;
  do {
    all.push_back(p);
  } while (std::next_permutation(p.begin(), p.end()));
  return all;
}

// Build CSR of the adjacent‐swap graph on `all` (size nv, perm length n)
static void build_csr(
    std::vector<Perm> const &all,
    std::unordered_map<Perm,int> const &inv,
    int n,
    std::vector<idx_t> &xadj,
    std::vector<idx_t> &adjncy)
{
  idx_t nv = all.size();
  xadj.resize(nv+1);
  adjncy.reserve(nv * (n-1));
  idx_t ec = 0;
  for (idx_t i = 0; i < nv; ++i) {
    xadj[i] = ec;
    auto const &v = all[i];
    for (int t = 1; t < n; ++t) {
      Perm u = v;
      std::swap(u[t-1], u[t]);
      adjncy.push_back(inv.at(u));
      ++ec;
    }
  }
  xadj[nv] = ec;
}

// METIS partition (called on rank 0)
static void metis_partition(
    const std::vector<idx_t> &xadj,
    const std::vector<idx_t> &adjncy,
    int world_size,
    std::vector<idx_t> &part)
{
  idx_t nv     = xadj.size() - 1;
  idx_t ncon   = 1;                 // number of balancing constraints
  idx_t nparts = world_size;       // how many parts to split into
  idx_t objval;

  part.resize(nv);

  int ret = METIS_PartGraphKway(
    /*nvtxs=*/ const_cast<idx_t*>(&nv),
    /*ncon=*/  &ncon,
    /*xadj=*/  const_cast<idx_t*>(xadj.data()),
    /*adjncy=*/const_cast<idx_t*>(adjncy.data()),
    /*vwgt=*/  nullptr,
    /*vsize=*/ nullptr,
    /*adjwgt=*/nullptr,
    /*nparts=*/&nparts,
    /*tpwgts=*/nullptr,
    /*ubvec=*/ nullptr,
    /*options=*/nullptr,
    /*objval=*/&objval,
    /*part=*/  part.data()
  );

  if (ret != METIS_OK) {
    std::cerr << "METIS_PartGraphKway failed with code " << ret << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

// Is this the root perm [1,2,…,n]?
static bool is_root(Perm const &v) {
  for (int i = 0; i < (int)v.size(); ++i)
    if (v[i] != i+1) return false;
  return true;
}

// Precompute for each local perm a map value→index
static std::unordered_map<Perm,std::unordered_map<int,int>>
precompute_positions(std::vector<Perm> const &verts)
{
  std::unordered_map<Perm,std::unordered_map<int,int>> pos;
  pos.reserve(verts.size());
  for (auto &v : verts) {
    std::unordered_map<int,int> m;
    m.reserve(v.size());
    for (int i = 0; i < (int)v.size(); ++i)
      m[v[i]] = i;
    pos.emplace(v, std::move(m));
  }
  return pos;
}

// Compute the parent of v in IST_t: swap t,t-1 in v
static Perm compute_parent(
    Perm const &v, int t,
    std::unordered_map<Perm,std::unordered_map<int,int>> const &pos)
{
  int idx = pos.at(v).at(t);
  if (idx == 0) return v;
  Perm p = v;
  std::swap(p[idx], p[idx-1]);
  return p;
}

int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world);
  if(argc!=3){
    if(rank==0) std::cerr<<"Usage: "<<argv[0]<<" n n_threads\n";
    MPI_Finalize();
    return 1;
  }
  int n = std::stoi(argv[1]);
  int nt = std::stoi(argv[2]);

  // 1) Rank 0 builds full graph & partitions
  std::vector<Perm> all;
  std::vector<idx_t> xadj, adjncy, part;
  if(rank==0){
    all = generate_all_perms(n);
    std::unordered_map<Perm,int> inv;
    inv.reserve(all.size());
    for(int i=0;i<(int)all.size();++i) inv.emplace(all[i],i);
    build_csr(all, inv, n, xadj, adjncy);
    metis_partition(xadj, adjncy, world, part);
  }

  // 2) Broadcast nv, CSR, part, and flat-perms
  idx_t nv=0; if(rank==0) nv=all.size();
  MPI_Bcast(&nv,1,MPI_LONG,0,MPI_COMM_WORLD);

  xadj.resize(nv+1);
  MPI_Bcast(xadj.data(), nv+1, MPI_LONG, 0, MPI_COMM_WORLD);

  idx_t eSz = 0;
  if(rank==0) eSz = adjncy.size();
  MPI_Bcast(&eSz,1,MPI_LONG,0,MPI_COMM_WORLD);
  adjncy.resize(eSz);
  MPI_Bcast(adjncy.data(), eSz, MPI_LONG, 0, MPI_COMM_WORLD);

  part.resize(nv);
  MPI_Bcast(part.data(), nv, MPI_LONG, 0, MPI_COMM_WORLD);

  // Perms flattened
  std::vector<int> flat;
  if(rank==0){
    flat.reserve(nv*n);
    for(auto &v: all)
      flat.insert(flat.end(), v.begin(), v.end());
  }
  flat.resize(nv*n);
  MPI_Bcast(flat.data(), nv*n, MPI_INT, 0, MPI_COMM_WORLD);

  // 3) Reconstruct local vertices
  std::vector<Perm> local;
  local.reserve(nv/world + 1);
  for(idx_t i=0;i<nv;++i){
    if(part[i]==rank){
      Perm v(n);
      std::copy(flat.begin()+i*n, flat.begin()+(i+1)*n, v.begin());
      local.push_back(v);
    }
  }

  // 4) Precompute positions and build ISTs in parallel
  auto pos = precompute_positions(local);

  // We'll collect each rank's block of lines into one string
  std::ostringstream oss;
  for(int t=1;t<n;++t){
    oss<<"T="<<t<<"\n";
    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for(int i=0;i<(int)local.size();++i){
      auto v = local[i];
      Perm p = is_root(v) ? v : compute_parent(v,t,pos);
      std::ostringstream line;
      line<<"(";
      for(int j=0;j<n;++j) line<<v[j]<<(j+1<n?", ":"");
      line<<") -> (";
      for(int j=0;j<n;++j) line<<p[j]<<(j+1<n?", ":"");
      line<<")\n";
      #pragma omp critical
      oss<<line.str();
    }
  }
  std::string myblock = oss.str();
  int mylen = myblock.size() + 1;  // include null

  // 5) Gather on rank 0
  std::vector<int> lengths;
  if(rank==0) lengths.resize(world);
  MPI_Gather(&mylen,1,MPI_INT,
             lengths.data(),1,MPI_INT,
             0,MPI_COMM_WORLD);

  std::vector<int> displs;
  std::string allblocks;
  if(rank==0){
    displs.resize(world);
    int sum=0;
    for(int r=0;r<world;++r){
      displs[r]=sum;
      sum += lengths[r];
    }
    allblocks.resize(sum);
  }

  MPI_Gatherv(myblock.c_str(), mylen, MPI_CHAR,
              rank==0?allblocks.data():nullptr,
              lengths.data(), displs.data(),
              MPI_CHAR, 0, MPI_COMM_WORLD);

  // 6) Rank 0 writes the single output file
  if(rank==0){
    fs::path out = fs::path("openMP_Bn")/std::to_string(n);
    fs::create_directories(out);
    std::ofstream ofs((out/("Bn"+std::to_string(n)
                           +"_ISTs_Seq_Parents.txt")).string());

    // Split each rank's block and emit per-tree sections
    std::vector<std::string> blocks(world);
    for(int r=0;r<world;++r){
      blocks[r] = allblocks.substr(displs[r], lengths[r]);
    }
    for(int t=1;t<n;++t){
      ofs<<"Tree t="<<t<<" (node \u2192 parent):\n";
      for(int r=0;r<world;++r){
        std::istringstream ib(blocks[r]);
        std::string line;
        bool inblock=false;
        while(std::getline(ib,line)){
          if(line.rfind("T=",0)==0){
            int bt = std::stoi(line.substr(2));
            inblock = (bt==t);
            continue;
          }
          if(inblock && !line.empty()){
            ofs<<line<<"\n";
          }
        }
      }
      ofs<<"\n";
    }
    std::cout<<"Wrote "<<(out/("Bn"+std::to_string(n)
                           +"_ISTs_Seq_Parents.txt")).string()<<"\n";
  }

  MPI_Finalize();
  return 0;
}
