#!/usr/bin/env python
"""Refactored QASM3 to MLIR converter generating QIR dialect code
Usage: python QASM2MLIR.py -i input.qasm -o output.mlir
"""

import argparse
import logging
import sys
from typing import Any, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure, Reset
from qiskit.circuit.library import (
    Barrier,
    CCXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.qasm2.parse import LEGACY_CUSTOM_INSTRUCTIONS

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


class QIRInitOp(MLIROperation):
    op_name = "init"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap) -> None:
        super().__init__(value_map)


class QIRSeedOp(MLIROperation):
    op_name = "seed"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, seed: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(seed)


class QIRHOp(MLIROperation):
    op_name = "H"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRXOp(MLIROperation):
    op_name = "X"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRYOp(MLIROperation):
    op_name = "Y"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRZOp(MLIROperation):
    op_name = "Z"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRCNOTOp(MLIROperation):
    op_name = "CNOT"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, ctl: SSAValue, tgt: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(ctl)
        self.add_operand(tgt)


class QIRCZOp(MLIROperation):
    op_name = "Cz"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, ctl: SSAValue, tgt: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(ctl)
        self.add_operand(tgt)


class QIRCCXOp(MLIROperation):
    op_name = "CCX"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, ctl1: SSAValue, ctl2: SSAValue, tgt: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(ctl1)
        self.add_operand(ctl2)
        self.add_operand(tgt)


class QIRRxOp(MLIROperation):
    op_name = "Rx"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(angle)


class QIRRyOp(MLIROperation):
    op_name = "Ry"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(angle)


class QIRRzOp(MLIROperation):
    op_name = "Rz"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(angle)


class QIRU2Op(MLIROperation):
    op_name = "U2"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, phi: SSAValue, lam: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(phi)
        self.add_operand(lam)


class QIRU1Op(MLIROperation):
    op_name = "U1"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, lam: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)
        self.add_operand(lam)


class QIRU3Op(MLIROperation):
    op_name = "U3"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue, theta: SSAValue, phi: SSAValue, lam: SSAValue) -> None:
        super().__init__(value_map)
        # we still emit a single U3 op if you prefer; QIR dialect supports it directly
        self.add_operand(qubit)
        self.add_operand(theta)
        self.add_operand(phi)
        self.add_operand(lam)


class QIRSwapOp(MLIROperation):
    op_name = "swap"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, lhs: SSAValue, rhs: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(lhs)
        self.add_operand(rhs)


class QIRSOp(MLIROperation):
    op_name = "S"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRSDGOp(MLIROperation):
    op_name = "Sdg"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRTOp(MLIROperation):
    op_name = "T"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRTDGOp(MLIROperation):
    op_name = "Tdg"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubit: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(qubit)


class QIRCRzOp(MLIROperation):
    op_name = "CRz"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, ctl: SSAValue, tgt: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(ctl)
        self.add_operand(tgt)
        self.add_operand(angle)


class QIRCRyOp(MLIROperation):
    op_name = "CRy"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, ctl: SSAValue, tgt: SSAValue, angle: SSAValue) -> None:
        super().__init__(value_map)
        self.add_operand(ctl)
        self.add_operand(tgt)
        self.add_operand(angle)


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


class BarrierOp(MLIROperation):
    op_name = "barrier"
    op_dialect = "qir"

    def __init__(self, value_map: SSAValueMap, qubits: list[SSAValue]) -> None:
        super().__init__(value_map)
        # no operands, no results
        # barrier takes any number of qubit operands
        for q in qubits:
            self.add_operand(q)


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
        # dispatch
        if isinstance(instr_op, HGate):
            return self.visit_h(instr_op, qargs, cargs)
        if isinstance(instr_op, XGate):
            return self.visit_x(instr_op, qargs, cargs)
        if isinstance(instr_op, YGate):
            return self.visit_y(instr_op, qargs, cargs)
        if isinstance(instr_op, ZGate):
            return self.visit_z(instr_op, qargs, cargs)
        if isinstance(instr_op, CXGate):
            return self.visit_cx(instr_op, qargs, cargs)
        if isinstance(instr_op, CZGate):
            return self.visit_cz(instr_op, qargs, cargs)
        if isinstance(instr_op, CCXGate):
            return self.visit_ccx(instr_op, qargs, cargs)
        if isinstance(instr_op, RXGate):
            return self.visit_rx(instr_op, qargs, cargs)
        if isinstance(instr_op, RYGate):
            return self.visit_ry(instr_op, qargs, cargs)
        if isinstance(instr_op, RZGate):
            return self.visit_rz(instr_op, qargs, cargs)
        if isinstance(instr_op, U3Gate):
            return self.visit_u3(instr_op, qargs, cargs)
        if isinstance(instr_op, U2Gate):
            return self.visit_u2(instr_op, qargs, cargs)
        if isinstance(instr_op, U1Gate):
            return self.visit_u1(instr_op, qargs, cargs)
        if isinstance(instr_op, SwapGate):
            return self.visit_swap(instr_op, qargs, cargs)
        if isinstance(instr_op, SGate):
            return self.visit_s(instr_op, qargs, cargs)
        if isinstance(instr_op, SdgGate):
            return self.visit_sdg(instr_op, qargs, cargs)
        if isinstance(instr_op, TGate):
            return self.visit_t(instr_op, qargs, cargs)
        if isinstance(instr_op, TdgGate):
            return self.visit_tdg(instr_op, qargs, cargs)
        if isinstance(instr_op, CRZGate):
            return self.visit_crz(instr_op, qargs, cargs)
        if isinstance(instr_op, CRYGate):
            return self.visit_cry(instr_op, qargs, cargs)
        if isinstance(instr_op, Measure):
            return self.visit_measure(instr_op, qargs, cargs)
        if isinstance(instr_op, Reset):
            return self.visit_reset(instr_op, qargs, cargs)
        if isinstance(instr_op, Barrier):
            return self.visit_barrier(instr_op, qargs, cargs)
        return self.generic_visit(instr_op, qargs, cargs)

    def generic_visit(self, instr_op: Instruction, qargs: list[Any], cargs: list[Any]) -> None:
        if getattr(instr_op, "definition", None) is not None:
            if instr_op.definition is None:
                raise UnimplementedError("Instruction definition is None")
            def_circ: QuantumCircuit = instr_op.definition
            # Build a map from the *definition’s* ephemeral qubits
            # back to the real qubits in this call site.
            mapping = {q_def: q_act for q_def, q_act in zip(def_circ.qubits, qargs)}

            # Now walk the defined-circuit’s own .data
            for instr in def_circ.data:
                # CircuitInstruction now has named attributes, not unpackable
                sub_inst = instr.operation
                sub_qargs = instr.qubits
                sub_cargs = instr.clbits

                # remap into our real-call-site qubits
                real_qargs = [mapping[q] for q in sub_qargs]
                self.visit(sub_inst, real_qargs, sub_cargs)

            return

        raise UnimplementedError(f"Unsupported operation: {type(instr_op).__name__}")

    def visit_h(self, instr_op: HGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRHOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_x(self, instr_op: XGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRXOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_y(self, instr_op: YGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRYOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_z(self, instr_op: ZGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRZOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_cx(self, instr_op: CXGate, qargs, cargs):
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        self.func.block.add_op(QIRCNOTOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2)))

    def visit_cz(self, instr_op: CZGate, qargs, cargs):
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        self.func.block.add_op(QIRCZOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2)))

    def visit_ccx(self, instr_op: CCXGate, qargs, cargs):
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        r3, i3 = self.qubit_info[qargs[2]]
        self.func.block.add_op(
            QIRCCXOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2), self.get_qubit(r3, i3))
        )

    def _emit_rotation(self, OpClass, instr_op, qargs, cargs):
        angle = float(instr_op.params[0])
        r, i = self.qubit_info[qargs[0]]
        const = ConstFloatOp(self.func.value_map, angle, MLIRType("f64"))
        self.func.block.add_op(const)
        self.func.block.add_op(OpClass(self.func.value_map, self.get_qubit(r, i), const.results[0]))

    def visit_rx(self, instr_op: RXGate, qargs, cargs):
        return self._emit_rotation(QIRRxOp, instr_op, qargs, cargs)

    def visit_ry(self, instr_op: RYGate, qargs, cargs):
        return self._emit_rotation(QIRRyOp, instr_op, qargs, cargs)

    def visit_rz(self, instr_op: RZGate, qargs, cargs):
        return self._emit_rotation(QIRRzOp, instr_op, qargs, cargs)

    def visit_u3(self, instr_op: U3Gate, qargs, cargs):
        theta, phi, lam = map(float, instr_op.params)
        r, i = self.qubit_info[qargs[0]]
        # constants
        cθ = ConstFloatOp(self.func.value_map, theta, MLIRType("f64"))
        cφ = ConstFloatOp(self.func.value_map, phi, MLIRType("f64"))
        cλ = ConstFloatOp(self.func.value_map, lam, MLIRType("f64"))
        self.func.block.add_op(cθ)
        self.func.block.add_op(cφ)
        self.func.block.add_op(cλ)
        self.func.block.add_op(QIRU3Op(self.func.value_map, self.get_qubit(r, i), cθ.results[0], cφ.results[0], cλ.results[0]))

    def visit_u2(self, instr_op: U2Gate, qargs, cargs):
        phi, lam = map(float, instr_op.params)
        r, i = self.qubit_info[qargs[0]]
        cφ = ConstFloatOp(self.func.value_map, phi, MLIRType("f64"))
        cλ = ConstFloatOp(self.func.value_map, lam, MLIRType("f64"))
        self.func.block.add_op(cφ)
        self.func.block.add_op(cλ)
        self.func.block.add_op(QIRU2Op(self.func.value_map, self.get_qubit(r, i), cφ.results[0], cλ.results[0]))

    def visit_u1(self, instr_op: U1Gate, qargs, cargs):
        lam = float(instr_op.params[0])
        r, i = self.qubit_info[qargs[0]]
        cλ = ConstFloatOp(self.func.value_map, lam, MLIRType("f64"))
        self.func.block.add_op(cλ)
        self.func.block.add_op(QIRU1Op(self.func.value_map, self.get_qubit(r, i), cλ.results[0]))

    def visit_s(self, instr_op: SGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRSOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_sdg(self, instr_op: SdgGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRSDGOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_t(self, instr_op: TGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRTOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_tdg(self, instr_op: TdgGate, qargs, cargs):
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRTDGOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_crz(self, instr_op: CRZGate, qargs, cargs):
        angle = float(instr_op.params[0])
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        cθ = ConstFloatOp(self.func.value_map, angle, MLIRType("f64"))
        self.func.block.add_op(cθ)
        self.func.block.add_op(QIRCRzOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2), cθ.results[0]))

    def visit_cry(self, instr_op: CRYGate, qargs, cargs):
        angle = float(instr_op.params[0])
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        cθ = ConstFloatOp(self.func.value_map, angle, MLIRType("f64"))
        self.func.block.add_op(cθ)
        self.func.block.add_op(QIRCRyOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2), cθ.results[0]))

    def visit_swap(self, instr_op: SwapGate, qargs: list[Any], cargs: list[Any]) -> None:
        r1, i1 = self.qubit_info[qargs[0]]
        r2, i2 = self.qubit_info[qargs[1]]
        self.func.block.add_op(QIRSwapOp(self.func.value_map, self.get_qubit(r1, i1), self.get_qubit(r2, i2)))

    def visit_reset(self, instr_op: Reset, qargs: list[Any], cargs: list[Any]) -> None:
        r, i = self.qubit_info[qargs[0]]
        self.func.block.add_op(QIRResetOp(self.func.value_map, self.get_qubit(r, i)))

    def visit_measure(self, instr_op: Measure, qargs, cargs):
        # Get SSAValues for qubit and its result buffer
        qr, qi = self.qubit_info[qargs[0]]
        qubit_ssa = self.get_qubit(qr, qi)
        cr, ci = self.clbit_info[cargs[0]]
        result_ssa = self.get_result(cr, ci)

        # Emit the measure (no results)
        measure_op = QIRMeasureOp(self.func.value_map, qubit_ssa, result_ssa)
        self.func.block.add_op(measure_op)

        # Now emit the read_measurement using the same result SSA
        read_op = QIRReadMeasurementOp(self.func.value_map, result_ssa)
        self.func.block.add_op(read_op)

    def visit_barrier(self, instr_op: Barrier, qargs: list[Any], cargs: list[Any]) -> None:
        # Emit a single barrier on all the qubits in qargs
        # Gather the SSAValues for each qubit argument
        qubit_ssas: list[SSAValue] = []
        for q in qargs:
            reg, idx = self.qubit_info[q]
            qubit_ssas.append(self.get_qubit(reg, idx))
        self.func.block.add_op(BarrierOp(self.func.value_map, qubit_ssas))


def QASMToMLIR(code: str) -> MLIRModule:
    try:
        circuit: QuantumCircuit = qasm2_loads(code, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as e:
        raise ConversionError(f"QASM2 parse failed: {e}")
    # *** DEBUG: print the decomposed circuit diagram ***
    # print("⟵ circuit:")
    # print(circuit.draw(output="text"))

    module = MLIRModule()
    func = MLIRFunction("main")

    qubit_info = {q: (qreg.name, i) for qreg in circuit.qregs for i, q in enumerate(qreg)}
    clbit_info = {c: (creg.name, i) for creg in circuit.cregs for i, c in enumerate(creg)}
    qubit_map, result_map = {}, {}

    def get_qubit(reg, idx):
        key = (reg, idx)
        if key not in qubit_map:
            alloc = QIRAllocOp(func.value_map)
            func.block.add_op(alloc)
            qubit_map[key] = alloc.results[0]
        return qubit_map[key]

    def get_result(reg, idx):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input QASM file")
    parser.add_argument("-o", "--output", help="Output MLIR file")
    args = parser.parse_args()
    code = open(args.input).read() if args.input else sys.stdin.read()
    module = QASMToMLIR(code)
    mlir = str(module)
    if args.output:
        open(args.output, "w").write(mlir)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
