#!/bin/bash

# Parameters to test
N_VALUES=(10000 50000)
K_VALUES=(4)
MPI_PROCS=(2 4)
OMP_THREADS=(2 4)

# Setup
mkdir -p logs

# ==== Sequential Python ====
echo "Running ist_seq.py..."
for n in "${N_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    python3 -m cProfile -o "logs/seq_n${n}_k${k}.prof" ist_seq.py <<< "$n"$'\n'"$k"
  done
done

# ==== MPI Python ====
echo "Running MPI.py..."
for n in "${N_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    for np in "${MPI_PROCS[@]}"; do
      echo "  MPI -np $np with n=$n, k=$k"
      mpirun -np "$np" python3 MPI.py <<< "$n"$'\n'"$k" > "logs/mpi_n${n}_k${k}_np${np}.log"
    done
  done
done

# ==== OpenMP+MPI C++ ====
echo "Compiling and running hybrid..."
tau_cxx.sh -openmp -mpi -o hybrid_inst openmp_c++.cpp

for n in "${N_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    for np in "${MPI_PROCS[@]}"; do
      for omp in "${OMP_THREADS[@]}"; do
        echo "  Hybrid MPI=$np, OMP=$omp, n=$n, k=$k"
        export OMP_NUM_THREADS="$omp"
        echo -e "$n\n$k" | mpirun -np "$np" tau_exec -T mpi,openmp ./hybrid_inst > "logs/hybrid_n${n}_k${k}_np${np}_omp${omp}.log"
      done
    done
  done
done

echo "All runs completed. See logs/"
