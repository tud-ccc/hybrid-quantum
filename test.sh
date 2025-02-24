#!/bin/bash

CINM=/net/media/scratch/quantum/hybrid-quantum/quantum/build/bin/quantum-opt
TRANS=/net/media/scratch/quantum/hybrid-quantum/quantum/build/bin/quantum-translate
total_tests=0
failed_tests=0

# Function to run test and check for errors
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

run_test "Basic Quantum Circuit" "$CINM /net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit.mlir "
run_test "QUANTUM TO LLVMIR" "$CINM -convert-func-to-llvm --convert-arith-to-llvm --convert-qir-to-llvm /net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit.mlir -o  /net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit_llvm.mlir --mlir-print-ir-after-all"
run_test "LLVMIR TO LLVM" "$TRANS  --mlir-to-llvmir /net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit_llvm.mlir -o  /net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit.ll --print-after-all"
run_test "QIR generator" "just qir quantum/examples/simple_circuit.ll"
run_test "QIR Runner" "/net/media/scratch/quantum/hybrid-quantum/quantum/examples/simple_circuit.out"

# # Print summary
# echo "Test Summary:" 
# echo "Tests failed: $failed_tests out of $total_tests"
