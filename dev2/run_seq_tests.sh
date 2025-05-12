#!/bin/bash

# Create results directory
RESULTS_DIR="seq_results"
mkdir -p $RESULTS_DIR

# Log file for all results
LOG_FILE="$RESULTS_DIR/seq_results.log"
echo "Sequential Test Results" > $LOG_FILE
echo "=====================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Function to run sequential test and log results
run_seq_test() {
    local n=$1
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "Running test: n=$n" | tee -a $LOG_FILE
    echo "Timestamp: $timestamp" | tee -a $LOG_FILE
    
    # Run sequential program and capture output
    output=$(python3 /home/fatima/Desktop/PDC_Project/dev2/ist_seq.py $n 2>&1)
    
    # Extract execution time
    exec_time=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
    
    # Log results
    echo "Execution time: $exec_time seconds" | tee -a $LOG_FILE
    echo "Output:" | tee -a $LOG_FILE
    echo "$output" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

# Run tests for n=2 to 7
for n in {2..7}; do
    run_seq_test $n
done

echo "All sequential tests completed. Results saved in $LOG_FILE" 