"""Refactored QASM3 to MLIR converter generating QIR dialect code
Usage: python QASM2MLIR.py -i input.qasm -o output.mlir
"""

import argparse
import logging
import sys
from typing import Any, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure, Reset
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import CXGate, HGate, RXGate, SwapGate, XGate
from qiskit.qasm3 import loads as qasm3_loads

# === Logging Setup ===


def setup_logger(level: int) -> logging.Logger:
    logger = logging.getLogger("mlir_converter")
    if getattr(logger, "_setup_done", False):
        return logger
    logger._setup_done = True  # type: ignore
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger(logging.WARNING)


class ConversionError(Exception):
    pass


class UnimplementedError(Exception):
    pass


class MLIRBase:
    def __str__(self) -> str:
        return "\n".join(self.serialize())

    def indent(self, lines: list[str], size: int = 1) -> list[str]:
        return ["  " * size + line for line in lines]

    def serialize(self) -> list[str]:
        raise UnimplementedError("serialize not implemented")


class MLIRType(MLIRBase):
    def __init__(self, name: str, dialect: str = "std") -> None:
        self.name: str = name
        self.dialect: str = dialect

    def serialize(self) -> list[str]:
        prefix = "" if self.dialect == "std" else f"!{self.dialect}."
        return [f"{prefix}{self.name}"]


class QubitType(MLIRType):
    def __init__(self) -> None:
        super().__init__(name="qubit", dialect="qir")


class ResultType(MLIRType):
    def __init__(self) -> None:
        super().__init__(name="result", dialect="qir")


class SSAValue(MLIRBase):
    def __init__(self, name: str, ty: MLIRType) -> None:
        self.name: str = name
        self.ty: MLIRType = ty

    def show(self) -> str:
        return f"%{self.name}"

    def serialize(self) -> list[str]:
        return [self.show()]


class SSAValueMap:
    def __init__(self) -> None:
        self.map: dict[str | tuple[str, int], SSAValue] = {}
        self.counter: int = 0

    def new_value(self, ty: MLIRType, label: str = "") -> SSAValue:
        name = str(self.counter)
        self.counter += 1
        val = SSAValue(name=name, ty=ty)
        if label:
            self.map[label] = val
        return val


class MLIROperation(MLIRBase):
    op_name: str
    op_dialect: str

    def __init__(self, value_map: SSAValueMap) -> None:
        self.value_map: SSAValueMap = value_map
        self.operands: list[SSAValue] = []
        self.results: list[SSAValue] = []
        if not hasattr(self, "op_name") or not hasattr(self, "op_dialect"):
            raise UnimplementedError("Operation must define op_name and op_dialect")

    def add_operand(self, op: SSAValue) -> None:
        self.operands.append(op)

    def add_result(self, ty: MLIRType) -> SSAValue:
        res = self.value_map.new_value(ty)
        self.results.append(res)
        return res

    def serialize(self) -> list[str]:
        full_name = f"{self.op_dialect}.{self.op_name}"
        res_str = ", ".join(r.show() for r in self.results)
        ops_str = ", ".join(o.show() for o in self.operands)
        in_types = ", ".join(o.ty.serialize()[0] for o in self.operands)
        out_types = ", ".join(r.ty.serialize()[0] for r in self.results)
        if res_str:
            return [f'{res_str} = "{full_name}" ({ops_str}) : ({in_types}) -> ({out_types})']
        return [f'"{full_name}" ({ops_str}) : ({in_types}) -> ()']


# QIR dialect operations


class QIRAllocOp(MLIROperation):
    op_name = "alloc"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap) -> None:
        super().__init__(value_map)
        self.add_result(QubitType())


class QIRResultAllocOp(MLIROperation):
    op_name = "ralloc"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap) -> None:
        super().__init__(value_map)
        self.add_result(ResultType())


class QIRHOp(MLIROperation):
    op_name = "H"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRCNOTOp(MLIROperation):
    op_name = "CNOT"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, control: SSAValue, target: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(control)
        self.add_operand(target)


class QIRMeasureOp(MLIROperation):
    op_name = "measure"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, result: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(result)


class QIRReadMeasurementOp(MLIROperation):
    op_name = "read_measurement"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, result_val: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(result_val)
        self.add_result(MLIRType(name="tensor<1xi1>", dialect="tensor"))


class QIRXOp(MLIROperation):
    op_name = "X"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRRxOp(MLIROperation):
    op_name = "Rx"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(angle)


class QIRSwapOp(MLIROperation):
    op_name = "swap"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, lhs: SSAValue, rhs: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(lhs)
        self.add_operand(rhs)


class QIRResetOp(MLIROperation):
    op_name = "reset"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class ConstFloatOp(MLIROperation):
    op_name = "constant"
    op_dialect = "arith"

    def __init__(self, value_map: SSAValueMap, value: float, result_type: MLIRType) -> None:
        super().__init__(value_map)
        self.const_value: float = value
        self.result_type: MLIRType = result_type
        self.add_result(self.result_type)

    def serialize(self) -> list[str]:
        name = self.results[0].show()
        ty = self.result_type.serialize()[0]
        return [f"{name} = {self.op_dialect}.{self.op_name} {self.const_value:.6f} : {ty}"]


class ArithCmpIOp(MLIROperation):
    op_name = "cmpi"
    op_dialect = "arith"

    def __init__(self, value_map: SSAValueMap, lhs: SSAValue, rhs: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(lhs)
        self.add_operand(rhs)
        self.add_result(MLIRType("i1", dialect="std"))


class SCFIfOp(MLIRBase):
    op_name = "if"
    op_dialect = "scf"

    def __init__(self, predicate: SSAValue, then_block: "MLIRBlock", else_block: Optional["MLIRBlock"] = None) -> None:
        self.predicate = predicate
        self.then_block = then_block
        self.else_block = else_block

    def serialize(self) -> list[str]:
        lines = [f"scf.if {self.predicate.show()} {{"]
        lines += self.indent(self.then_block.serialize())
        if self.else_block:
            lines.append("} else {")
            lines += self.indent(self.else_block.serialize())
        lines.append("}")
        return lines


class ReturnOp(MLIROperation):
    op_name = "return"
    op_dialect = "std"

    def __init__(self, value_map: SSAValueMap) -> None:
        super().__init__(value_map)

    def serialize(self) -> list[str]:
        return ["return"]


class MLIRBlock(MLIRBase):
    def __init__(self, value_map: SSAValueMap) -> None:
        self.value_map: SSAValueMap = value_map
        self.ops: list[MLIRBase] = []

    def add_op(self, op: MLIRBase) -> None:
        self.ops.append(op)

    def serialize(self) -> list[str]:
        lines: list[str] = []
        for op in self.ops:
            lines.extend(op.serialize())
        return lines


class MLIRFunction(MLIRBase):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_map: SSAValueMap = SSAValueMap()
        self.block: MLIRBlock = MLIRBlock(self.value_map)

    def serialize(self) -> list[str]:
        header = f"func.func @{self.name}() {{"
        body = self.indent(self.block.serialize())
        return [header] + body + ["}"]


class MLIRModule(MLIRBase):
    def __init__(self) -> None:
        self.functions: list[MLIRFunction] = []

    def add_function(self, fn: MLIRFunction) -> None:
        self.functions.append(fn)

    def serialize(self) -> list[str]:
        lines: list[str] = ["module {"]
        for fn in self.functions:
            lines.extend(self.indent(fn.serialize()))
        lines.append("}")
        return lines


# Visitor class


class QASMToMLIRVisitor:
    def __init__(
        self,
        func: MLIRFunction,
        qubit_info: dict[Any, tuple[str, int]],
        clbit_info: dict[Any, tuple[str, int]],
        get_qubit: Any,
        get_result: Any,
    ) -> None:
        self.func = func
        self.qubit_info = qubit_info
        self.clbit_info = clbit_info
        self.get_qubit = get_qubit
        self.get_result = get_result

    def visit(self, instr_op: Instruction, qargs: list[Any], cargs: list[Any]) -> None:
        # explicit class‐based dispatch:
        if isinstance(instr_op, HGate):
            return self.visit_h(instr_op, qargs, cargs)
        if isinstance(instr_op, XGate):
            return self.visit_x(instr_op, qargs, cargs)
        if isinstance(instr_op, CXGate):
            return self.visit_cx(instr_op, qargs, cargs)
        if isinstance(instr_op, RXGate):
            return self.visit_rx(instr_op, qargs, cargs)
        if isinstance(instr_op, SwapGate):
            return self.visit_swap(instr_op, qargs, cargs)
        if isinstance(instr_op, Reset):
            return self.visit_reset(instr_op, qargs, cargs)
        if isinstance(instr_op, Measure):
            return self.visit_measure(instr_op, qargs, cargs)
        if isinstance(instr_op, IfElseOp):
            return self.visit_if_else(instr_op, qargs, cargs)
        return self.generic_visit(instr_op, qargs, cargs)

    def generic_visit(self, instr_op: Instruction, qargs: list[Any], cargs: list[Any]) -> None:
        raise UnimplementedError(f"Unsupported operation: {type(instr_op).__name__}")

    def visit_h(self, instr_op: HGate, qargs: list[Any], cargs: list[Any]) -> None:
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRHOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_cx(self, instr_op: CXGate, qargs: list[Any], cargs: list[Any]) -> None:
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        self.func.block.add_op(
            QIRCNOTOp(
                self.func.value_map,
                self.get_qubit(r1, i1),
                self.get_qubit(r2, i2),
            )
        )

    def visit_measure(self, instr_op: Measure, qargs: list[Any], cargs: list[Any]) -> None:
        qr, qi = self.qubit_info[qargs[0]]
        cr, ci = self.clbit_info[cargs[0]]
        self.func.block.add_op(QIRMeasureOp(self.func.value_map, self.get_qubit(qr, qi), self.get_result(cr, ci)))
        self.func.block.add_op(QIRReadMeasurementOp(self.func.value_map, self.get_result(cr, ci)))

    def visit_x(self, instr_op: XGate, qargs: list[Any], cargs: list[Any]) -> None:
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRXOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_rx(self, instr_op: RXGate, qargs: list[Any], cargs: list[Any]) -> None:
        angle = float(instr_op.params[0])
        r, i = self.qubit_info[qargs[0]]
        const = ConstFloatOp(self.func.value_map, angle, MLIRType("f64"))
        self.func.block.add_op(const)
        self.func.block.add_op(QIRRxOp(self.func.value_map, self.get_qubit(r, i), const.results[0]))

    def visit_swap(self, instr_op: SwapGate, qargs: list[Any], cargs: list[Any]) -> None:
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        self.func.block.add_op(QIRSwapOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2)))

    def visit_reset(self, instr_op: Reset, qargs: list[Any], cargs: list[Any]) -> None:
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRResetOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_if_else(self, instr_op: IfElseOp, qargs: list[Any], cargs: list[Any]) -> None:
        # only simple (bit, val) conditions supported
        cond = instr_op.condition
        if isinstance(cond, tuple):
            cond_bit, cond_val = cond
        else:
            raise ConversionError(f"Unsupported condition type: {type(cond).__name__}")

        creg, cidx = self.clbit_info[cond_bit]
        bit_ssa = self.get_result(creg, cidx)

        const = ConstFloatOp(self.func.value_map, 1.0 if cond_val else 0.0, MLIRType("i1"))
        self.func.block.add_op(const)
        cmpi = ArithCmpIOp(self.func.value_map, bit_ssa, const.results[0])
        self.func.block.add_op(cmpi)

        # build the regions
        then_circ = instr_op.blocks[0]
        else_circ = instr_op.blocks[1] if len(instr_op.blocks) > 1 else None
        then_block = MLIRBlock(self.func.value_map)
        else_block = MLIRBlock(self.func.value_map) if else_circ else None

        for nested in then_circ.data:
            saved = self.func.block
            self.func.block = then_block
            self.visit(nested.operation, nested.qubits, nested.clbits)
            self.func.block = saved

        if else_circ:
            for nested in else_circ.data:
                saved = self.func.block
                self.func.block = else_block  # type: ignore
                self.visit(nested.operation, nested.qubits, nested.clbits)
                self.func.block = saved

        scf_if = SCFIfOp(cmpi.results[0], then_block, else_block)
        self.func.block.add_op(scf_if)


def QASMToMLIR(code: str) -> MLIRModule:
    try:
        circuit: QuantumCircuit = qasm3_loads(code)
    except Exception as e:
        raise ConversionError(f"QASM3 parse failed: {e}")

    module = MLIRModule()
    func = MLIRFunction("main")

    qubit_info = {q: (qreg.name, i) for qreg in circuit.qregs for i, q in enumerate(qreg)}
    clbit_info = {c: (creg.name, i) for creg in circuit.cregs for i, c in enumerate(creg)}
    qubit_map: dict[tuple[str, int], SSAValue] = {}
    result_map: dict[tuple[str, int], SSAValue] = {}

    def get_qubit(reg: str, idx: int) -> SSAValue:
        key = (reg, idx)
        if key not in qubit_map:
            alloc = QIRAllocOp(func.value_map)
            func.block.add_op(alloc)
            qubit_map[key] = alloc.results[0]
        return qubit_map[key]

    def get_result(reg: str, idx: int) -> SSAValue:
        key = (reg, idx)
        if key not in result_map:
            ralloc = QIRResultAllocOp(func.value_map)
            func.block.add_op(ralloc)
            result_map[key] = ralloc.results[0]
        return result_map[key]

    visitor = QASMToMLIRVisitor(func, qubit_info, clbit_info, get_qubit, get_result)
    for instr in circuit.data:
        visitor.visit(instr.operation, instr.qubits, instr.clbits)

    func.block.add_op(ReturnOp(func.value_map))
    module.add_function(func)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="QASM3 to MLIR Converter")
    parser.add_argument("-i", "--input", help="Input QASM file")
    parser.add_argument("-o", "--output", help="Output MLIR file")
    args = parser.parse_args()

    code = open(args.input).read() if args.input else sys.stdin.read()
    try:
        module = QASMToMLIR(code)
    except ConversionError as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

    mlir = str(module)
    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
