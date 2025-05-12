#!/bin/bash

# === Environment setup ===
export PATH=$PATH:$HOME/Desktop/PDC_Project/dev2/tau_install/tau_install/tau-2.34/x86_64/bin
export TAU_MAKEFILE=$HOME/Desktop/PDC_Project/dev2/tau_install/tau_install/tau-2.34/x86_64/lib/Makefile.tau-mpi-openmp

# === TAU configuration ===
export TAU_PROFILE=1
export TAU_TRACE=0
export TAU_METRICS="TIME,PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM"
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=10
export TAU_VERBOSE=1
export TAU_GRAPHVIZ=1

# === Check for gnuplot ===
command -v gnuplot >/dev/null 2>&1 || { echo >&2 "gnuplot is required but not installed. Aborting."; exit 1; }

TAU_PPROF=$HOME/Desktop/PDC_Project/dev2/tau_install/tau_install/tau-2.34/x86_64/bin/pprof

mkdir -p analysis_results/scalability/{strong_scaling,weak_scaling}
mkdir -p analysis_results/scalability/{strong_scaling,weak_scaling}/{profiles,visualizations,data}

# === Compile ===
echo "Compiling with TAU..."
tau_cxx.sh -fopenmp -o openmp_cpp_inst openmp_c++.cpp

# === Function: Strong Scaling with Partitioning Efficiency ===
run_strong_scaling() {
    local n=$1 k=$2 max_procs=$3 threads=$4
    echo "Running strong scaling: n=$n, k=$k, max_procs=$max_procs, threads=$threads"
    export OMP_NUM_THREADS=$threads

    local run_dir="analysis_results/scalability/strong_scaling/profiles/n${n}_k${k}_t${threads}"
    mkdir -p "$run_dir"
    declare -A exec_times

    for procs in $(seq 1 $max_procs); do
        echo "  -> $procs processes"
        rm -f profile.*

        local log_file="analysis_results/scalability/strong_scaling/data/n${n}_k${k}_p${procs}_t${threads}.log"
        mpirun -np $procs tau_exec -T mpi,openmp ./openmp_cpp_inst $n $k > "$log_file" 2>&1

        mv profile.* "$run_dir/" 2>/dev/null || true
        exec_times[$procs]=$(grep "Total Execution Time:" "$log_file" | awk '{print $4}')
    done

    echo "Generating strong scaling plot..."

    local csv_path="analysis_results/scalability/strong_scaling/strong_scaling_n${n}_k${k}_t${threads}.csv"
    local plot_path="analysis_results/scalability/strong_scaling/visualizations/strong_scaling_n${n}_k${k}_t${threads}.png"

    echo "Processes,ExecutionTime,Speedup,Efficiency" > "$csv_path"
    local valid_points=0

    for procs in $(seq 1 $max_procs); do
        if [ -n "${exec_times[1]}" ] && [ -n "${exec_times[$procs]}" ]; then
            local speedup=$(echo "scale=4; ${exec_times[1]}/${exec_times[$procs]}" | bc)
            local efficiency=$(echo "scale=4; $speedup / $procs" | bc)
            echo "$procs,${exec_times[$procs]},$speedup,$efficiency" >> "$csv_path"
            ((valid_points++))
        else
            echo "Skipping $procs processes: missing execution time"
        fi
    done

    if [ "$valid_points" -ge 2 ]; then
        gnuplot <<EOF
set terminal png size 1000,600
set output "$plot_path"
set title "Strong Scaling with Partitioning Efficiency (n=$n, k=$k, threads=$threads)"
set xlabel "Processes"
set ylabel "Speedup"
set y2label "Partitioning Efficiency"
set y2tics
set datafile separator ","
set key outside
plot "$csv_path" using 1:3 with linespoints title "Speedup", \
     "$csv_path" using 1:4 axes x1y2 with linespoints title "Partitioning Efficiency"
EOF
    else
        echo "Not enough valid points to generate strong scaling plot for n=$n, k=$k, threads=$threads"
    fi
}

# === Function: Weak Scaling ===
run_weak_scaling() {
    local base_n=$1 k=$2 max_procs=$3 threads=$4
    echo "Running weak scaling: base_n=$base_n, k=$k, max_procs=$max_procs, threads=$threads"
    export OMP_NUM_THREADS=$threads

    local run_dir="analysis_results/scalability/weak_scaling/profiles/k${k}_omp${threads}"
    mkdir -p "$run_dir"
    declare -A exec_times

    for procs in $(seq 1 $max_procs); do
        local n=$((base_n * procs))
         if [ "$n" -eq 8 ]; then
            local n=7
            #echo "  -> Skipping $procs processes (n=$n is excluded)"
            #continue
        fi
        echo "  -> $procs processes, n=$n"
        rm -f profile.*

        local log_file="analysis_results/scalability/weak_scaling/data/n${n}_k${k}_p${procs}_t${threads}.log"
        mpirun -np $procs tau_exec -T mpi,openmp ./openmp_cpp_inst $n $k > "$log_file" 2>&1

        mv profile.* "$run_dir/" 2>/dev/null || true
        exec_times[$procs]=$(grep "Total Execution Time:" "$log_file" | awk '{print $4}')
    done

    echo "Generating weak scaling plot..."

    local csv_path="analysis_results/scalability/weak_scaling/weak_scaling_k${k}_t${threads}.csv"
    local plot_path="analysis_results/scalability/weak_scaling/visualizations/weak_scaling_k${k}_t${threads}.png"

    echo "Processes,ExecutionTime,Efficiency" > "$csv_path"
    local valid_points=0
    for procs in $(seq 1 $max_procs); do
        if [ -n "${exec_times[1]}" ] && [ -n "${exec_times[$procs]}" ]; then
            local eff=$(echo "scale=4; ${exec_times[1]}/${exec_times[$procs]}" | bc)
            echo "$procs,${exec_times[$procs]},$eff" >> "$csv_path"
            ((valid_points++))
        fi
    done

    if [ "$valid_points" -ge 2 ]; then
        gnuplot <<EOF
set terminal png
set output "$plot_path"
set title "Weak Scaling (k=$k, threads=$threads)"
set xlabel "Processes"
set ylabel "Efficiency"
set datafile separator ","
plot "$csv_path" using 1:3 with linespoints title "Efficiency"
EOF
    else
        echo "Not enough valid points to generate weak scaling plot for k=$k, threads=$threads"
    fi
}

# === Run strong scaling ===
echo "Running strong scaling tests..."
for n in 5 6 7; do
    for k in 3 4 5; do
        for threads in 2 4 6; do
            run_strong_scaling $n $k 4 $threads
        done
    done
done

# === Run weak scaling ===
echo "Running weak scaling tests..."
for base_n in 2; do
    for k in 2 3 4; do
        for threads in 2 4 6; do
            max_procs=4
            if [ $max_procs -ge 2 ]; then
                run_weak_scaling $base_n $k $max_procs $threads
            else
                echo "Skipping weak scaling: base_n=$base_n leads to insufficient data points (max_procs=$max_procs)"
            fi
        done
    done
done
