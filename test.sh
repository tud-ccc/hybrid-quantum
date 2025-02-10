#!/bin/bash

CINM=/net/media/scratch/quantum/Cinnamon/quantum/build/bin/quantum-opt
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
#run_test "Basic Quantum Circuit" "cinm-opt /net/media/scratch/quantum/Cinnamon/quantum/examples/conversion.mlir --quantum-torch"

# Testing a basic transformation pass for gate cancellation operation
#run_test "Function test" " /net/media/scratch/quantum/Cinnamon/quantum/examples/circuit.mlir"

# Testing all quantum gates operation from Ops.td file
#run_test "Quantum Gates Operations" "cinm-opt /net/media/scratch/quantum/Cinnamon/quantum/examples/circuit.mlir"

# Testing conversion of quantum dialect to LLVM IR
#run_test "Quantum Parsing" "cinm-opt /net/media/scratch/quantum/Cinnamon/quantum/examples/all_operations.mlir"

run_test "Basic Quantum Circuit" "$CINM /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit.mlir "
run_test "QUANTUM TO LLVMIR" "$CINM -convert-func-to-llvm --convert-arith-to-llvm --convert-qir-to-llvm /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit.mlir -o  /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit_llvm.mlir"
run_test "LLVMIR TO LLVM" "$TRANS  --mlir-to-llvmir /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit_llvm.mlir -o  /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit.ll"
#run_test "QIR RUNNER" "qir-runner  --file /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit.ll -opaque-pointers"

# run_test "Quantum to LLVM Conversion" "cinm-opt \
#   --convert-scf-to-cf \
#   --convert-cf-to-llvm \
#   --convert-arith-to-llvm \
#   --convert-func-to-llvm \
#   --reconcile-unrealized-casts \
#   /net/media/scratch/quantum/Cinnamon/quantum/examples/all_operations.mlir"


#   run_test "Quantum to LLVM Conversion" "cinm-opt \
#   --convert-arith-to-llvm \
#   --convert-func-to-llvm \
#   --finalize-memref-to-llvm \
#   --convert-quantum-to-llvm \
#   /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit_llvm.mlir"

#run_test "LLVM IR to LLVM Conversion" "cinm-translate --mlir-to-llvmir /net/media/scratch/quantum/Cinnamon/quantum/examples/simple_circuit.mlir -o /net/media/scratch/quantum/Cinnamon/quantum/examples/output.ll"

# run_test "QIR RUNNER" "qir-runner --file /net/media/scratch/quantum/Cinnamon/quantum/examples/output.ll"

#run_test "Torch2Quantum" "cinm-opt  --allow-unregistered-dialect /net/media/scratch/quantum/Cinnamon/quantum/examples/conversion.mlir --quantum-torch"

# # Print summary
# echo "Test Summary:" 
# echo "Tests failed: $failed_tests out of $total_tests"
