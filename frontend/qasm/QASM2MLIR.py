# QASM3 to MLIR converter generating QIR dialect code
# Usage: python QASM2MLIR.py -i input.qasm -o output.mlir

import sys
import logging
import argparse
from typing import Union, Tuple, Dict

from qiskit.qasm3 import loads as qasm3_loads
from qiskit import QuantumCircuit

# === Logging Setup ===


def setupLogger(lev):
    logger = logging.getLogger('mlir_converter')
    if 'logger_setup' in globals():
        return logger
    global logger_setup
    logger_setup = True
    logger.setLevel(lev)
    ch = logging.StreamHandler()
    ch.setLevel(lev)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setupLogger(logging.WARNING)

# === Exceptions ===


class ConversionError(Exception):
    pass


class UnimplementedError(Exception):
    pass

# === Minimal MLIR Infrastructure ===


class MLIRBase:
    def __str__(self): return "\n".join(self.serialize())
    def indent(self, lines, size=1): return [
        "  " * size + line for line in lines]

    def serialize(self): raise UnimplementedError("serialize not implemented")


class MLIRType(MLIRBase):
    def __init__(self, name, dialect="std"):
        self.name = name
        self.dialect = dialect

    def serialize(self):
        prefix = "" if self.dialect == "std" else f"!{self.dialect}."
        return [f"{prefix}{self.name}"]


class QubitType(MLIRType):
    def __init__(self): super().__init__("qubit", dialect="qir")


class ResultType(MLIRType):
    def __init__(self): super().__init__("result", dialect="qir")


class SSAValue(MLIRBase):
    def __init__(self, name, ty):
        self.name = name
        self.ty = ty

    def show(self): return f"%{self.name}"
    def serialize(self): return [self.show()]


class SSAValueMap:
    def __init__(self):
        self.map: Dict[Union[str, Tuple[str, int]], SSAValue] = {}
        self.counter = 0

    def new_value(self, ty, label=""):
        name = str(self.counter)
        self.counter += 1
        val = SSAValue(name, ty)
        if label:
            self.map[label] = val
        return val


class MLIROperation(MLIRBase):
    def __init__(self, value_map: SSAValueMap):
        self.value_map = value_map
        self.operands = []
        self.results = []

    def add_operand(self, op): self.operands.append(op)

    def add_result(self, ty):
        res = self.value_map.new_value(ty)
        self.results.append(res)
        return res

    def serialize(self):
        res_str = ", ".join([r.show() for r in self.results])
        ops_str = ", ".join([o.show() for o in self.operands])
        return [f"{res_str} = \"{self.opname}\" ({ops_str}) : ({', '.join([o.ty.serialize()[0] for o in self.operands])}) -> ({', '.join([r.ty.serialize()[0] for r in self.results])})"] if res_str else [f"\"{self.opname}\" ({ops_str}) : ({', '.join([o.ty.serialize()[0] for o in self.operands])}) -> ()"]


class QIRAllocOp(MLIROperation):
    opname = "qir.alloc"

    def __init__(self, value_map: SSAValueMap):
        super().__init__(value_map)
        self.add_result(QubitType())

    def serialize(self): return [
        f"{self.results[0].show()} = \"{self.opname}\" () : () -> (!qir.qubit)"]


class QIRResultAllocOp(MLIROperation):
    opname = "qir.ralloc"

    def __init__(self, value_map: SSAValueMap):
        super().__init__(value_map)
        self.add_result(ResultType())

    def serialize(self): return [
        f"{self.results[0].show()} = \"{self.opname}\" () : () -> (!qir.result)"]


class QIRHOp(MLIROperation):
    opname = "qir.H"

    def __init__(self, value_map: SSAValueMap, qubit):
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRCNOTOp(MLIROperation):
    opname = "qir.CNOT"

    def __init__(self, value_map: SSAValueMap, control, target):
        super().__init__(value_map)
        self.add_operand(control)
        self.add_operand(target)


class QIRMeasureOp(MLIROperation):
    opname = "qir.measure"

    def __init__(self, value_map: SSAValueMap, qubit, result):
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(result)


class QIRReadMeasurementOp(MLIROperation):
    opname = "qir.read_measurement"

    def __init__(self, value_map: SSAValueMap, result_value: SSAValue):
        super().__init__(value_map)
        self.add_operand(result_value)
        measurement_ty = MLIRType("tensor<1xi1>")
        self.add_result(measurement_ty)


class QIRXOp(MLIROperation):
    opname = "qir.X"

    def __init__(self, value_map: SSAValueMap, qubit):
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRRxOp(MLIROperation):
    opname = "qir.Rx"

    def __init__(self, value_map: SSAValueMap, qubit, angle_value):
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(angle_value)


class QIRSwapOp(MLIROperation):
    opname = "qir.swap"

    def __init__(self, value_map: SSAValueMap, lhs, rhs):
        super().__init__(value_map)
        self.add_operand(lhs)
        self.add_operand(rhs)


class QIRResetOp(MLIROperation):
    opname = "qir.reset"

    def __init__(self, value_map: SSAValueMap, qubit):
        super().__init__(value_map)
        self.add_operand(qubit)


class ConstFloatOp(MLIROperation):
    opname = "arith.constant"

    def __init__(self, value_map: SSAValueMap, const_value: float, result_type: MLIRType):
        super().__init__(value_map)
        self.const_value = const_value
        self.result_type = result_type
        self.add_result(self.result_type)

    def serialize(self):
        result = self.results[0].show()
        type_str = self.result_type.serialize()[0]
        return [f"{result} = {self.opname} {self.const_value:.6f} : {type_str}"]


class ReturnOp(MLIROperation):
    opname = "return"

    def __init__(self, value_map: SSAValueMap):
        super().__init__(value_map)

    def serialize(self):
        return [f"{self.opname}"]


class MLIRBlock(MLIRBase):
    def __init__(self, value_map: SSAValueMap):
        self.value_map = value_map
        self.ops = []

    def add_op(self, op: MLIROperation): self.ops.append(op)
    def serialize(self): return [
        line for op in self.ops for line in op.serialize()]


class MLIRFunction(MLIRBase):
    def __init__(self, name):
        self.name = name
        self.value_map = SSAValueMap()
        self.block = MLIRBlock(self.value_map)

    def serialize(self):
        header = [f"func.func @{self.name}() {{"]
        body = self.block.serialize()
        return header + self.indent(body) + ["}"]


class MLIRModule(MLIRBase):
    def __init__(self): self.functions = []
    def add_function(self, func: MLIRFunction): self.functions.append(func)

    def serialize(self):
        lines = ["module {"] + self.indent(
            [line for func in self.functions for line in func.serialize()]) + ["}"]
        return lines

# === QASM to MLIR Conversion ===


def QASMToMLIR(code: str) -> MLIRModule:
    try:
        circuit: QuantumCircuit = qasm3_loads(code)
    except Exception as e:
        raise ConversionError(f"QASM3 parse failed: {e}")

    module = MLIRModule()
    func = MLIRFunction("main")

    # Create reverse mapping from physical qubits/clbits to register info
    qubit_info_map = {qubit: (qreg.name, idx)
                      for qreg in circuit.qregs
                      for idx, qubit in enumerate(qreg)}
    clbit_info_map = {clbit: (creg.name, idx)
                      for creg in circuit.cregs
                      for idx, clbit in enumerate(creg)}

    # Allocate qubits and results
    qubit_map = {}
    result_map = {}
    for qreg in circuit.qregs:
        for idx in range(qreg.size):
            alloc_op = QIRAllocOp(func.value_map)
            func.block.add_op(alloc_op)
            qubit_map[(qreg.name, idx)] = alloc_op.results[0]

    for creg in circuit.cregs:
        for idx in range(creg.size):
            result_alloc_op = QIRResultAllocOp(func.value_map)
            func.block.add_op(result_alloc_op)
            result_map[(creg.name, idx)] = result_alloc_op.results[0]

    # Process instructions using modern API
    for instruction in circuit.data:
        operation = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        gate_name = operation.name.lower()

        # Handle Hadamard
        if gate_name == "h":
            regname, idx = qubit_info_map[qargs[0]]
            qubit = qubit_map[(regname, idx)]
            op = QIRHOp(func.value_map, qubit)
            func.block.add_op(op)

        # Handle CNOT
        elif gate_name == "cx":
            ctrl_reg, ctrl_idx = qubit_info_map[qargs[0]]
            tgt_reg, tgt_idx = qubit_info_map[qargs[1]]
            control = qubit_map[(ctrl_reg, ctrl_idx)]
            target = qubit_map[(tgt_reg, tgt_idx)]
            op = QIRCNOTOp(func.value_map, control, target)
            func.block.add_op(op)

        # Handle Measurement
        elif gate_name == "measure":
            q_reg, q_idx = qubit_info_map[qargs[0]]
            c_reg, c_idx = clbit_info_map[cargs[0]]
            qubit = qubit_map[(q_reg, q_idx)]
            result = result_map[(c_reg, c_idx)]
            meas_op = QIRMeasureOp(func.value_map, qubit, result)
            func.block.add_op(meas_op)
            read_op = QIRReadMeasurementOp(func.value_map, result)
            func.block.add_op(read_op)

        # Handle X gate
        elif gate_name == "x":
            reg, idx = qubit_info_map[qargs[0]]
            qubit = qubit_map[(reg, idx)]
            op = QIRXOp(func.value_map, qubit)
            func.block.add_op(op)

        # Handle RX gate
        elif gate_name == "rx":
            angle = float(operation.params[0])
            reg, idx = qubit_info_map[qargs[0]]
            qubit = qubit_map[(reg, idx)]
            angle_op = ConstFloatOp(
                func.value_map, angle, MLIRType("f64", dialect="std"))
            func.block.add_op(angle_op)
            op = QIRRxOp(func.value_map, qubit, angle_op.results[0])
            func.block.add_op(op)

        # Handle SWAP gate
        elif gate_name == "swap":
            lhs_reg, lhs_idx = qubit_info_map[qargs[0]]
            rhs_reg, rhs_idx = qubit_info_map[qargs[1]]
            lhs = qubit_map[(lhs_reg, lhs_idx)]
            rhs = qubit_map[(rhs_reg, rhs_idx)]
            op = QIRSwapOp(func.value_map, lhs, rhs)
            func.block.add_op(op)

        # Handle reset operation
        elif gate_name == "reset":
            reg, idx = qubit_info_map[qargs[0]]
            qubit = qubit_map[(reg, idx)]
            op = QIRResetOp(func.value_map, qubit)
            func.block.add_op(op)

        else:
            raise UnimplementedError(f"Unsupported operation: {gate_name}")

    func.block.add_op(ReturnOp(func.value_map))
    module.add_function(func)
    return module

# === Main Function ===


def main():
    parser = argparse.ArgumentParser(
        description="QASM3 to MLIR Converter (Qiskit 1.2+ compatible, QIR dialect)")
    parser.add_argument("-i", dest="input",
                        help="Input QASM file", required=False)
    parser.add_argument("-o", dest="output",
                        help="Output MLIR file", required=False)
    args = parser.parse_args()

    code = open(args.input, "r").read() if args.input else sys.stdin.read()

    try:
        module = QASMToMLIR(code)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

    mlir_code = str(module)
    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_code)
    else:
        print(mlir_code)


if __name__ == "__main__":
    main()
