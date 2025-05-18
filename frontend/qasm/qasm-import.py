#!/usr/bin/env python3
"""
#   Frontend generating QIR dialect code from QASM2 and QASM3 code.
#   Usage: `python qasm-import.py -i input.qasm -o output.mlir`
#
# @author  Washim Neupane (washim.neupane@outlook.com)
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from enum import Enum
from typing import Union

sys.path.append("/Users/lschuetze/git/hybrid-quantum/build/python_packages/quantum-mlir")
from mlir.dialects import func, qir
from mlir.dialects.builtin import Block
from mlir.ir import Context, InsertionPoint, Location, Module, StringAttr, TypeAttr, Value
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Operation, Qubit
from qiskit.circuit import library as lib
from qiskit.circuit.classical.expr import Expr
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.qasm2.parse import LEGACY_CUSTOM_INSTRUCTIONS
from qiskit.qasm2.parse import _DefinedGate as QASM2_Gate


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


class ConversionError(RuntimeError): ...


class ParseError(RuntimeError): ...


# Specify the QASM version the frontend conforms to
class QASMVersion(Enum):
    Unspecified = 0
    QASM_2_0 = 1
    QASM_3_0 = 2


# Types that can be coerced to a valid Qubit specifier in a circuit.
QubitSpecifier = Union[
    Qubit,
    QuantumRegister,
    # int,
    # slice,
    # Sequence[Union[Qubit, int]],
]
# Types that can be coerced to a valid Clbit specifier in a circuit.
ClbitSpecifier = Union[
    Clbit,
    ClassicalRegister,
    # int,
    # slice,
    # Sequence[Union[Clbit, int]],
]


class Scope:
    def __init__(self, visited: dict[str, qir.GateOp] | None = None) -> None:
        self.qregs: dict[str, qir.AllocOp] = {}
        self.cregs: dict[str, qir.AllocResultOp] = {}
        self.visitedGates: dict[str, qir.GateOp] = visited if visited is not None else {}

    @classmethod
    def fromList(
        cls, qregs: list[QubitSpecifier], cregs: list[ClbitSpecifier], visited: dict[str, qir.GateOp] | None = None
    ) -> Scope:
        s = cls(visited)
        s.qregs = {str(q): None for qreg in qregs for q in qreg}
        s.cregs = {str(c): None for creg in cregs for c in creg}
        return s

    @classmethod
    def fromMap(
        cls,
        qregs: dict[str, qir.AllocOp],
        cregs: dict[str, qir.AllocOp],
        visited: dict[str, qir.GateOp] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = qregs
        s.cregs = cregs
        return s

    def findAlloc(self, reg: QubitSpecifier) -> Value:
        return self.qregs.get(str(reg))

    def setAlloc(self, reg: QubitSpecifier, alloc: Value) -> None:
        self.qregs[str(reg)] = alloc

    def findResult(self, reg: ClbitSpecifier) -> Value:
        return self.qregs.get(str(reg))

    def setResult(self, reg: ClbitSpecifier, ralloc: Value) -> None:
        self.qregs[str(reg)] = ralloc

    def findGate(self, gate: QASM2_Gate) -> qir.GateOp:
        return self.visitedGates.get(str(gate.name))

    def setGate(self, gate: QASM2_Gate, newGate: qir.GateOp) -> None:
        self.visitedGates[str(gate.name)] = newGate


class QASMToMLIRVisitor:
    def __init__(self, compat: QASMVersion, context: Context, module: Module, loc: Location, block: Block, scope: Scope) -> None:
        self.compat: QASMVersion = compat
        self.context: Context = context
        self.module: Module = module
        self.loc: Location = loc
        self.block: Block = block
        self.scope = scope

    @classmethod
    def fromParent(cls, parent: QASMToMLIRVisitor, *, block: Block | None = None, scope: Scope | None = None):
        return cls(
            parent.compat,
            parent.context,
            parent.module,
            parent.loc,
            parent.block if block is None else block,
            parent.scope if scope is None else scope,
        )

    def visitCircuit(self, circuit: QuantumCircuit) -> None:
        for instr in circuit.data:
            if isinstance(instr.operation, Expr):
                self.visitClassic(instr)
            elif isinstance(instr.operation, Instruction):
                self.visitInstruction(instr.operation, instr.qubits, instr.clbits)
            elif isinstance(instr.operation, Operation):
                self.visitQuantum(instr.operation)
            else:
                raise ParseError(f"Unknown instruction: {instr} of type {type(instr)}")

    def visitQuantumRegister(self, reg: QuantumRegister) -> Value:
        if self.scope.findAlloc(reg) is None:
            # q: qir.QubitType = qir.QubitType.get(self.context)
            alloc: qir.AllocOp = qir.AllocOp(loc=self.loc)
            self.block.append(alloc)
            self.scope.setAlloc(reg, alloc.result)
            return alloc.result

        return self.scope.findAlloc(reg)

    def visitClassicalRegister(self, reg: ClassicalRegister) -> Value:
        if self.scope.findResult(reg) is None:
            # q: qir.QubitType = qir.ResultType.get(self.context)
            ralloc: qir.AllocResultOp = qir.AllocResultOp(loc=self.loc)
            self.block.append(ralloc)
            self.scope.setResult(reg, ralloc.result)
            return ralloc.result

        return self.scope.findResult(reg)

    def visitClassic(self, expr: Expr) -> None:
        raise NotImplementedError(f"Classic expressions are not supported for {expr}")

    # Operation encapsulates virtual instructions that must
    # be synthesized to physical instructions
    def visitQuantum(self, instr: Operation) -> None:
        raise NotImplementedError(f"Virtual quantum expressions are not supported for {instr}")

    # Instruction represents physical quantum instructions
    def visitInstruction(self, instr: Instruction, qubits: list[QuantumRegister], clbits: list[ClassicalRegister]) -> None:
        if isinstance(instr, lib.XGate):
            target: qir.QubitType = self.visitQuantumRegister(qubits[0])
            xgate: qir.XOp = qir.XOp(target, loc=self.loc, ip=InsertionPoint(self.block))
        elif isinstance(instr, lib.CXGate):
            control: qir.QubitType = self.visitQuantumRegister(qubits[0])
            target: qir.QubitType = self.visitQuantumRegister(qubits[1])
            cxgate: qir.CNOTOp = qir.CNOTOp(control, target, loc=self.loc, ip=InsertionPoint(self.block))
        elif isinstance(instr, lib.CCXGate):
            control1: qir.QubitType = self.visitQuantumRegister(qubits[0])
            control2: qir.QubitType = self.visitQuantumRegister(qubits[1])
            target: qir.QubitType = self.visitQuantumRegister(qubits[2])
            cxgate: qir.CCXOp = qir.CCXOp(control1, control2, target, loc=self.loc, ip=InsertionPoint(self.block))
        elif isinstance(instr, lib.Measure):
            qubit: qir.QubitType = self.visitQuantumRegister(qubits[0])
            bit: qir.ResultType = self.visitClassicalRegister(clbits[0])
            measureOp: qir.MeasureOp = qir.MeasureOp(qubit, bit, loc=self.loc, ip=InsertionPoint(self.block))
            # readMeasurement: qir.ReadMeasurementOp = qir.ReadMeasurementOp(measureOp.result, loc=self.loc, ip=InsertionPoint(self.block))
        elif isinstance(instr, QASM2_Gate):
            self._visitDefinedGate(instr, qubits, clbits)
        else:
            raise NotImplementedError(f"{instr} of type {type(instr)}")

    def _visitDefinedGate(self, instr: QASM2_Gate, qubits: list[QuantumRegister], clbits: list[ClassicalRegister]) -> None:
        if instr.definition is not None:
            if self.scope.findGate(instr) is None:
                # Construct qir.GateOp for dDefined custom gate
                # Insert into module body and recursively visit gate body
                inputs: list[qir.QubitType] = [qir.QubitType.get(self.context) for _ in range(instr.num_qubits)]
                gty: func.FunctionType = func.FunctionType.get(inputs=inputs, results=[], context=self.context)
                gate: qir.GateOp = qir.GateOp(
                    StringAttr.get(str(instr.name)),
                    TypeAttr.get(gty),
                    loc=self.loc,
                    ip=InsertionPoint.at_block_begin(self.module.body),
                )
                arg_locs = [self.loc for _ in inputs]
                gate.body.blocks.append(*inputs, arg_locs=arg_locs)
                gateBody: Block = gate.body.blocks[0]
                self.scope.setGate(instr, gate)

                circuit: QuantumCircuit = instr.definition
                gateQregs = {str(q): a for q, a in zip(circuit.qubits, gate.body.blocks[0].arguments)}
                innerGateScope: Scope = Scope.fromMap(gateQregs, {}, self.scope.visitedGates)
                visitor: QASMToMLIRVisitor = QASMToMLIRVisitor.fromParent(self, block=gateBody, scope=innerGateScope)
                visitor.visitCircuit(circuit)
            # Construct qir.CallOp for defined custom gate
            callee: StringAttr = StringAttr.get(str(instr.name))
            operands: list[qir.QubitType] = [self.visitQuantumRegister(q) for q in qubits]
            gateCall: qir.GateCallOp = qir.GateCallOp(callee, operands, loc=self.loc, ip=InsertionPoint(self.block))
        else:
            ParseError(f"Expected gate with definition but got {instr}")


def qasm_version(code: str) -> QASMVersion:
    match = re.search(r"OPENQASM\s+(\d+)\.(\d+);", code)

    if match:
        major = int(match.group(1))
        minor = int(match.group(2))

        if major == 2 and minor == 0:
            return QASMVersion.QASM_2_0
        elif major == 3 and minor == 0:
            return QASMVersion.QASM_3_0

    return QASMVersion.Unspecified


def QASMToMLIR(code: str) -> Module:
    compat: QASMVersion = qasm_version(code)
    circuit: QuantumCircuit

    match compat:
        case QASMVersion.QASM_2_0:
            try:
                circuit = qasm2_loads(code, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
            except Exception as e:
                raise ConversionError(f"QASM2 parse failed: {e}")
        case QASMVersion.QASM_3_0:
            raise NotImplementedError("QASM3 not implemented")
            # circuit = qasm3_loads(code)
        case QASMVersion.Unspecified:
            raise ParseError("No version string found")

    context: Context = Context()
    context.allow_unregistered_dialects = True
    qir.register_dialect(context)

    with context:
        location: Location = Location.unknown()
        module: Module = Module.create(location)

        qasm_main: func.FuncOp = func.FuncOp("qasm_main", ([], []), visibility="private", loc=location)
        qasm_main.add_entry_block()
        module.body.append(qasm_main)

        scope: Scope = Scope.fromList(circuit.qregs, circuit.cregs)
        visitor: QASMToMLIRVisitor = QASMToMLIRVisitor(compat, context, module, location, qasm_main.entry_block, scope)
        visitor.visitCircuit(circuit)

        func.ReturnOp([], loc=location, ip=InsertionPoint(qasm_main.entry_block))

    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input QASM file")
    parser.add_argument("-o", "--output", help="Output MLIR file")
    args = parser.parse_args()

    code: str = open(args.input).read() if args.input else sys.stdin.read()

    module: Module = QASMToMLIR(code)
    mlir: str = str(module)

    if args.output:
        open(args.output, "w").write(mlir)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
