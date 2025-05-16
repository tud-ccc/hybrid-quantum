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
from mlir.dialects import builtin, func, qir
from mlir.ir import Context, Location, Module, Value
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Gate, Instruction, Operation, Qubit
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
    def __init__(self, visited: list[QASM2_Gate] | None = None) -> None:
        self.qregs: dict[QubitSpecifier, qir.AllocOp] = {}
        self.cregs: dict[ClbitSpecifier, qir.AllocResultOp] = {}
        self.visitedGates: list[QASM2_Gate] = visited if visited is not None else []

    @classmethod
    def fromList(cls, qregs: list[QubitSpecifier], cregs: list[ClbitSpecifier], visited: list[QASM2_Gate] | None = None) -> Scope:
        s = cls(visited)
        s.qregs = {q: None for qreg in qregs for q in qreg}
        s.cregs = {c: None for creg in cregs for c in creg}
        return s

    @classmethod
    def fromMap(
        cls,
        qregs: dict[QubitSpecifier, qir.AllocOp],
        cregs: dict[ClbitSpecifier, qir.AllocOp],
        visited: list[QASM2_Gate] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = {q: None for qreg in qregs for q in qreg}
        s.cregs = {c: None for creg in cregs for c in creg}
        return s

    def findAlloc(self, reg: QubitSpecifier) -> qir.AllocOp:
        return self.qregs[reg]

    def setAlloc(self, reg: QubitSpecifier, alloc: qir.AllocOp) -> None:
        self.qregs[reg] = alloc

    def findResult(self, reg: ClbitSpecifier) -> qir.AllocResultOp:
        return self.qregs[reg]

    def setResult(self, reg: ClbitSpecifier, ralloc: qir.AllocResultOp) -> None:
        self.qregs[reg] = ralloc

    def hasVisited(self, gate: QASM2_Gate) -> bool:
        if gate not in self.visitedGates:
            self.visitedGates.append(gate)
            return False
        return True


class QASMToMLIRVisitor:
    def __init__(
        self, compat: QASMVersion, context: Context, module: builtin.Module, loc: Location, main: func.FuncOp, scope: Scope
    ) -> None:
        self.compat: QASMVersion = compat
        self.context: Context = context
        self.module: builtin.Module = module
        self.loc: Location = loc
        self.func: func.FuncOp = main
        self.scope = scope

    @classmethod
    def fromParent(cls, parent: QASMToMLIRVisitor):
        return cls(parent.compat, parent.context, parent.module, parent.loc, parent.func, parent.scope)

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
            q: qir.QubitType = qir.QubitType.get(self.context)
            alloc: qir.AllocOp = qir.AllocOp(q, loc=self.loc)
            self.func.entry_block.append(alloc)
            self.scope.setAlloc(reg, alloc)
            return alloc.result

        return self.scope.findAlloc(reg).result

    def visitClassicalRegister(self, reg: ClassicalRegister) -> Value:
        if self.scope.findResult(reg) is None:
            q: qir.QubitType = qir.ResultType.get(self.context)
            ralloc: qir.AllocResultOp = qir.AllocResultOp(q, loc=self.loc)
            self.func.entry_block.append(ralloc)
            self.scope.setResult(reg, ralloc)
            return ralloc.result

        return self.scope.findResult(reg).result

    def visitClassic(self, expr: Expr) -> None:
        raise NotImplementedError(f"Classic expressions are not supported for {expr}")

    # Operation encapsulates virtual instructions that must
    # be synthesized to physical instructions
    def visitQuantum(self, instr: Operation) -> None:
        raise NotImplementedError(f"Virtual quantum expressions are not supported for {instr}")

    # Instruction represents physical quantum instructions
    def visitInstruction(self, instr: Instruction, qubits: list[QuantumRegister], classic: list[ClassicalRegister]) -> None:
        if isinstance(instr, lib.XGate):
            q0 = self.visitQuantumRegister(qubits[0])
            xgate: qir.XOp = qir.XOp(q0, loc=self.loc)
            self.func.entry_block.append(xgate)
            return
        elif isinstance(instr, QASM2_Gate):
            if instr.definition is not None:
                if not self.scope.hasVisited(instr):
                    # TODO: create qir.gate in module and start writing to its body
                    gate: QuantumCircuit = instr.definition
                    visitor: QASMToMLIRVisitor = QASMToMLIRVisitor.fromParent(self)
                    visitor.visitCircuit(gate)
                # TODO: build call to gate in current func
                # qir.call(gate.name, values...)
                #
            else:
                ParseError(f"Expected gate with definition but got {instr}")
        else:
            raise NotImplementedError(f"{instr} of type {type(instr)}")

    def _visitGate(self, gate: Gate) -> None:
        pass


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


def QASMToMLIR(code: str) -> builtin.ModuleOp:
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
        visitor: QASMToMLIRVisitor = QASMToMLIRVisitor(compat, context, module, location, qasm_main, scope)
        visitor.visitCircuit(circuit)

    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input QASM file")
    parser.add_argument("-o", "--output", help="Output MLIR file")
    args = parser.parse_args()

    code: str = open(args.input).read() if args.input else sys.stdin.read()

    module: builtin.ModuleOp = QASMToMLIR(code)
    mlir: str = str(module)

    if args.output:
        open(args.output, "w").write(mlir)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
