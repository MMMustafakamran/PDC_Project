#!/bin/bash

# Create results directory
RESULTS_DIR="mpi_results"
mkdir -p $RESULTS_DIR

# Log file for all results
LOG_FILE="$RESULTS_DIR/mpi_results.log"
echo "MPI Test Results" > $LOG_FILE
echo "===============" >> $LOG_FILE
echo "" >> $LOG_FILE

# Function to run MPI and log results
run_mpi_test() {
    local n=$1
    local k=$2
    local procs=$3
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "Running test: n=$n, k=$k, processes=$procs" | tee -a $LOG_FILE
    echo "Timestamp: $timestamp" | tee -a $LOG_FILE
    
    # Run MPI and capture output
    output=$(mpiexec --hostfile hostfile -n $procs python3 /home/fatima/Desktop/PDC_Project/dev2/MPI.py $n $k 2>&1)
    
    # Extract execution time
    exec_time=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
    
    # Log results
    echo "Execution time: $exec_time seconds" | tee -a $LOG_FILE
    echo "Output:" | tee -a $LOG_FILE
    echo "$output" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

# Run tests with different combinations
for n in {2..7}; do
    for k in {2..4}; do
        for procs in {2..4}; do
            # Only run if k <= n (as k represents number of partitions)
            if [ $k -le $n ]; then
                run_mpi_test $n $k $procs
            fi
        done
    done
done

echo "All tests completed. Results saved in $LOG_FILE" 