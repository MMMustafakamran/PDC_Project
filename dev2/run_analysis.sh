#!/bin/bash

# Environment setup
export PATH=$PATH:$HOME/Desktop/PDC_Project/dev2/tau_install/tau_install/tau-2.34/x86_64/bin
export TAU_MAKEFILE=$HOME/Desktop/PDC_Project/dev2/tau_install/tau_install/tau-2.34/x86_64/lib/Makefile.tau-mpi-openmp

export TAU_PROFILE=1
export TAU_TRACE=0
export TAU_METRICS=TIME
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=10
export TAU_VERBOSE=1
export TAU_GRAPHVIZ=1  # Needed for PNG output

# Parameters
N_VALUES=(3 4 5 6)
K_VALUES=(2 3 4)
MPI_PROCS=(1 2 4)
OMP_THREADS=(1 2 4)

# Directories
mkdir -p logs
mkdir -p profiles
rm -f profile.*

# Compile
echo "Compiling with TAU..."
tau_cxx.sh -fopenmp -o openmp_cpp_inst openmp_c++.cpp

# Run tests
for n in "${N_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    for np in "${MPI_PROCS[@]}"; do
      for omp in "${OMP_THREADS[@]}"; do
        echo "Running: n=$n, k=$k, MPI=$np, OMP=$omp"
        export OMP_NUM_THREADS="$omp"
        mpirun -np "$np" tau_exec -T mpi,openmp ./openmp_cpp_inst "$n" "$k" > "logs/app_n${n}_k${k}_np${np}_omp${omp}.log" 2>&1

        run_dir="profiles/n${n}_k${k}_np${np}_omp${omp}"
        mkdir -p "$run_dir"
        mv profile.* "$run_dir/" 2>/dev/null || true

        cd "$run_dir" || continue

        echo "Generating visualizations..."
        prof_file="profile.0.0.0"

        if [ -f "$prof_file" ]; then
          pprof -flat -png -output flat.png "$prof_file"
          pprof -callgraph -png -output callgraph.png "$prof_file"
          pprof -time -png -output time_breakdown.png "$prof_file"
          pprof -time "$prof_file" > profile_time_report.txt
        else
          echo "Warning: $prof_file not found, skipping visualization."
        fi

        cd ../../
      done
    done
  done
done

echo "All runs completed. Check logs/ and profiles/ for results."
