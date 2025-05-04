// openmp_metis.cpp
//mpicxx -O3 -march=native -std=c++17 -fopenmp openmp_metis.cpp -lmetis -o openmp_metis
//mpirun -np 4 ./openmp_metis 5 4
// openmp_metis.cpp

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
using Perm = std::vector<int>;

namespace std {
  template<> struct hash<Perm> {
    size_t operator()(Perm const& v) const noexcept {
      size_t h = 0;
      for (int x : v) {
        h ^= std::hash<int>()(x) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
      }
      return h;
    }
  };
}

static std::vector<Perm> generate_all_perms(int n) {
  Perm p(n);
  for (int i = 0; i < n; ++i) p[i] = i+1;
  std::vector<Perm> all;
  do { all.push_back(p); }
  while (std::next_permutation(p.begin(), p.end()));
  return all;
}

static void build_csr(
    const std::vector<Perm>& all,
    const std::unordered_map<Perm,int>& inv,
    int n,
    std::vector<int>& xadj,
    std::vector<int>& adjncy)
{
  int nv = all.size();
  xadj.resize(nv+1);
  adjncy.reserve(nv * (n-1));
  int ec = 0;
  for (int i = 0; i < nv; ++i) {
    xadj[i] = ec;
    const auto& v = all[i];
    for (int t = 1; t < n; ++t) {
      Perm u = v;
      std::swap(u[t-1], u[t]);
      adjncy.push_back(inv.at(u));
      ++ec;
    }
  }
  xadj[nv] = ec;
}

static void metis_partition(
    const std::vector<int>& xadj,
    const std::vector<int>& adjncy,
    int world_size,
    std::vector<int>& part)
{
  int nv     = xadj.size() - 1;
  int ncon   = 1;
  int nparts = world_size;
  int objval;

  part.resize(nv);
  int ret = METIS_PartGraphKway(
    &nv, &ncon,
    const_cast<int*>(xadj.data()),
    const_cast<int*>(adjncy.data()),
    nullptr, nullptr, nullptr,
    &nparts,
    nullptr, nullptr,
    nullptr, &objval,
    part.data()
  );
  if (ret != METIS_OK) {
    std::cerr<<"METIS error "<<ret<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);
  }
}

static bool is_root(const Perm& v) {
  for (int i = 0; i < (int)v.size(); ++i)
    if (v[i] != i+1) return false;
  return true;
}

static std::unordered_map<Perm,std::unordered_map<int,int>>
precompute_positions(const std::vector<Perm>& verts)
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

static Perm compute_parent(
    const Perm& v, int t,
    const std::unordered_map<Perm,std::unordered_map<int,int>>& pos)
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
  int n  = std::stoi(argv[1]);
  int nt = std::stoi(argv[2]);

  // 1) Partition on rank 0
  std::vector<Perm> all;
  std::vector<int> xadj, adjncy, part;
  if(rank==0){
    all = generate_all_perms(n);
    std::unordered_map<Perm,int> inv;
    inv.reserve(all.size());
    for(int i=0;i<(int)all.size();++i) inv.emplace(all[i],i);
    build_csr(all, inv, n, xadj, adjncy);
    metis_partition(xadj, adjncy, world, part);
  }

  // 2) Broadcast as int / MPI_INT
  int nv = 0;
  if(rank==0) nv = all.size();
  MPI_Bcast(&nv,1,MPI_INT,0,MPI_COMM_WORLD);

  xadj.resize(nv+1);
  MPI_Bcast(xadj.data(), nv+1, MPI_INT, 0, MPI_COMM_WORLD);

  int eSz = 0;
  if(rank==0) eSz = adjncy.size();
  MPI_Bcast(&eSz,1,MPI_INT,0,MPI_COMM_WORLD);
  adjncy.resize(eSz);
  MPI_Bcast(adjncy.data(), eSz, MPI_INT, 0, MPI_COMM_WORLD);

  part.resize(nv);
  MPI_Bcast(part.data(), nv, MPI_INT, 0, MPI_COMM_WORLD);

  // broadcast flat perms
  std::vector<int> flat;
  if(rank==0){
    flat.reserve(nv*n);
    for(auto &v: all)
      flat.insert(flat.end(), v.begin(), v.end());
  }
  flat.resize(nv*n);
  MPI_Bcast(flat.data(), nv*n, MPI_INT, 0, MPI_COMM_WORLD);

  // 3) Build local list
  std::vector<Perm> local;
  for(int i=0;i<nv;++i){
    if(part[i]==rank){
      Perm v(n);
      std::copy(flat.begin()+i*n, flat.begin()+(i+1)*n, v.begin());
      local.push_back(v);
    }
  }

  // 4) Compute ISTs with OpenMP
  auto pos = precompute_positions(local);
  std::ostringstream oss;
  for(int t=1;t<n;++t){
    oss<<"T="<<t<<"\n";
    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for(int i=0;i<(int)local.size();++i){
      auto v = local[i];
      Perm p = is_root(v)?v:compute_parent(v,t,pos);
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
  int mylen = myblock.size()+1;

  // 5) Gather onto rank 0
  std::vector<int> lengths(world), displs(world);
  MPI_Gather(&mylen,1,MPI_INT,
             lengths.data(),1,MPI_INT,
             0,MPI_COMM_WORLD);
  int total = 0;
  if(rank==0){
    for(int i=0;i<world;++i){
      displs[i] = total;
      total += lengths[i];
    }
  }
  std::string allblocks;
  if(rank==0) allblocks.resize(total);
  MPI_Gatherv(myblock.c_str(), mylen, MPI_CHAR,
              rank==0?allblocks.data():nullptr,
              lengths.data(), displs.data(),
              MPI_CHAR, 0, MPI_COMM_WORLD);

  // 6) Write on rank 0
  if(rank==0){
    fs::path out = fs::path("openMP_Bn")/std::to_string(n);
    fs::create_directories(out);
    std::ofstream ofs((out/("Bn"+std::to_string(n)
                           +"_ISTs_Seq_Parents.txt")).string());
    std::vector<std::string> blocks(world);
    for(int r=0;r<world;++r)
      blocks[r] = allblocks.substr(displs[r], lengths[r]);
    for(int t=1;t<n;++t){
      ofs<<"Tree t="<<t<<" (node \u2192 parent):\n";
      for(int r=0;r<world;++r){
        std::istringstream ib(blocks[r]);
        std::string line;
        bool inblock=false;
        while(std::getline(ib,line)){
          if(line.rfind("T=",0)==0){
            inblock = (std::stoi(line.substr(2)) == t);
            continue;
          }
          if(inblock && !line.empty()) ofs<<line<<"\n";
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
