#!/bin/bash

# Paths to your tools
CINM=/net/media/scratch/quantum/hybrid-quantum/quantum/build/bin/quantum-opt
TRANS=/net/media/scratch/quantum/hybrid-quantum/quantum/build/bin/quantum-translate

# Directory containing the input files
INPUT_DIR="/net/media/scratch/quantum/hybrid-quantum/quantum/examples/qir"

# Variables to track test results
total_tests=0
failed_tests=0

# Function to run a test and check for errors
run_test() {
    local test_name="$1"
    local command="$2"
    
    ((total_tests++))
    
    echo "Running test: $test_name"
    output=$($command 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Success"
        echo "$output"
    else
        echo "Error in $test_name (Exit code: $exit_code):"
        echo "$output"
        ((failed_tests++))
    fi
    echo "------------------------"
}

# Function to process the input file
process_file() {
    local input_file="$1"
    local full_path="${INPUT_DIR}/${input_file}"
    local base_name="${input_file%.mlir}"  # Remove .mlir extension
    local llvm_mlir_file="${INPUT_DIR}/${base_name}_llvm.mlir"
    local llvm_ir_file="${INPUT_DIR}/${base_name}_llvm.ll"
    local output_file="${INPUT_DIR}/${base_name}_llvm.out"

    # Check if the input file exists
    if [[ ! -f "$full_path" ]]; then
        echo "Error: File $full_path does not exist."
        exit 1
    fi

    # Step 1: Lower QUANTUM to LLVMIR
    run_test "QUANTUM TO LLVMIR" \
        "$CINM \
          --convert-scf-to-cf \
          --canonicalize \
          --finalize-memref-to-llvm \
          --convert-func-to-llvm   \
          --convert-arith-to-llvm  \
          --convert-cf-to-llvm     \
          --convert-index-to-llvm  \
          --convert-qir-to-llvm \
          --reconcile-unrealized-casts \
           $full_path -o $llvm_mlir_file"

    # Step 2: Convert LLVMIR to LLVM
    run_test "LLVMIR TO LLVM" \
        "$TRANS --mlir-to-llvmir $llvm_mlir_file -o $llvm_ir_file"

    # Step 3: Generate QIR
    run_test "QIR generator" \
        "just qir $llvm_ir_file"

    # Step 4: Run the QIR program
    run_test "QIR Runner" \
        "$output_file"
}

# Parse command-line arguments
if [[ $# -ne 2 || "$1" != "-f" ]]; then
    echo "Usage: $0 -f <filename.mlir>"
    exit 1
fi

input_file="$2"
process_file "$input_file"

# Print summary
echo "Test Summary:"
echo "Total tests: $total_tests"
echo "Failed tests: $failed_tests"