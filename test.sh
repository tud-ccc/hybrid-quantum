#!/bin/bash

CINM=/net/media/scratch/quantum/Cinnamon/quantum/build/bin/cinm-opt
TRANS=/net/media/scratch/quantum/Cinnamon/quantum/build/bin/cinm-translate
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

# Basic circuit using Quantum dialect
#run_test "Basic Quantum Circuit" "$CINM  --allow-unregistered-dialect /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/conversion.mlir --quantum-torch"

# Testing a basic transformation pass for gate cancellation operation
#run_test "Function test" "$CINM /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/circuit.mlir"

# Testing all quantum gates operation from Ops.td file
#run_test "Quantum Gates Operations" "$CINM /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/circuit.mlir"

# Testing conversion of quantum dialect to LLVM IR
#run_test "Quantum Parsing" "$CINM /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/all_operations.mlir"

run_test "Basic Quantum Circuit" "$CINM  /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit.mlir "
run_test "QUANTUM TO LLVMIR" "$CINM --convert-arith-to-llvm --convert-quantum-to-llvm /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit.mlir -o  /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit_llvm.mlir"
run_test "LLVMIR TO LLVM" "$TRANS  --mlir-to-llvmir /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit_llvm.mlir -o  /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit.ll"
run_test "QIR RUNNER" "qir-runner  --file /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit.ll"

# run_test "Quantum to LLVM Conversion" "$CINM \
#   --convert-scf-to-cf \
#   --convert-cf-to-llvm \
#   --convert-arith-to-llvm \
#   --convert-func-to-llvm \
#   --reconcile-unrealized-casts \
#   /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/all_operations.mlir"


#   run_test "Quantum to LLVM Conversion" "$CINM \
#   --convert-arith-to-llvm \
#   --convert-func-to-llvm \
#   --finalize-memref-to-llvm \
#   --convert-quantum-to-llvm \
#   /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit_llvm.mlir"

#run_test "LLVM IR to LLVM Conversion" "cinm-translate --mlir-to-llvmir /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/simple_circuit.mlir -o /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/output.ll"

# run_test "QIR RUNNER" "qir-runner --file /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/output.ll"

#run_test "Torch2Quantum" "$CINM  --allow-unregistered-dialect /net/media/scratch/quantum/Cinnamon/quantum/test/Dialect/Quantum/conversion.mlir --quantum-torch"

# # Print summary
# echo "Test Summary:" 
# echo "Tests failed: $failed_tests out of $total_tests"
