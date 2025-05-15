#!/usr/bin/env python3
"""
#   Frontend generating QIR dialect code from QASM2 and QASM3 code.
#   Usage: `python qasm-import.py -i input.qasm -o output.mlir`
#
# @author  Washim Neupane (washim.neupane@outlook.com)
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

import argparse
import logging
import re
import sys
from enum import Enum

sys.path.append("/Users/lschuetze/git/hybrid-quantum/build/python_packages/quantum-mlir")
from mlir.dialects import builtin, func, qir
from mlir.ir import Context, Location, Module
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Operation
from qiskit.circuit import library as lib
from qiskit.circuit.classical.expr import Expr
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.qasm2.parse import LEGACY_CUSTOM_INSTRUCTIONS


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


"""
# Types that can be coerced to a valid Qubit specifier in a circuit.
QubitSpecifier = Union[
    Qubit,
    QuantumRegister,
    int,
    slice,
    Sequence[Union[Qubit, int]],
]
# Types that can be coerced to a valid Clbit specifier in a circuit.
ClbitSpecifier = Union[
    Clbit,
    ClassicalRegister,
    int,
    slice,
    Sequence[Union[Clbit, int]],
]
"""


class QASMToMLIRVisitor:
    def __init__(
        self,
        compat: QASMVersion,
        context: Context,
        module: builtin.Module,
        loc: Location,
        main: func.FuncOp,
        qregs: list[QuantumRegister],
        cregs: list[ClassicalRegister],
    ) -> None:
        self.compat: QASMVersion = compat
        self.context: Context = context
        self.module: builtin.Module = module
        self.loc: Location = loc
        self.func: func.FuncOp = main

        self.qalloc: dict[QuantumRegister, qir.AllocOp] = {q: None for qreg in qregs for q in qreg}
        self.ralloc: dict[ClassicalRegister, qir.ResultAllocOp] = {c: None for creg in cregs for c in creg}

    def visitQuantumRegister(self, reg: QuantumRegister) -> qir.AllocOp:
        if self.qalloc[reg] is None:
            q: qir.QubitType = qir.QubitType.get(self.context)
            self.qalloc[reg] = q.result
        return self.qalloc[reg]

    def visitClassicalRegister(self, reg: ClassicalRegister) -> None:
        pass

    def visitClassic(self, expr: Expr) -> None:
        raise NotImplementedError(f"Classic expressions are not supported for {expr}")

    # Operation encapsulates virtual instructions that must
    # be (physically) synthesized
    def visitQuantum(self, instr: Operation) -> None:
        raise NotImplementedError(f"Virtual quantum expressions are not supported for {instr}")

    # Instruction represents physical quantum instructions
    def visitInstruction(self, instr: Instruction, qubits: list[QuantumRegister], classic: list[ClassicalRegister]) -> None:
        if isinstance(instr, lib.XGate):
            q0 = self.visitQuantumRegister(qubits[0])
            xgate: qir.XOp = qir.XOp(None)
            self.func.body.append(xgate)

        raise NotImplementedError(f"{instr}")

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
        module.body.append(qasm_main)

        visitor: QASMToMLIRVisitor = QASMToMLIRVisitor(compat, context, module, location, qasm_main, circuit.qregs, circuit.cregs)

        for instr in circuit.data:
            if isinstance(instr.operation, Expr):
                visitor.visitClassic(instr)
            elif isinstance(instr.operation, Instruction):
                visitor.visitInstruction(instr.operation, instr.qubits, instr.clbits)
            elif isinstance(instr.operation, Operation):
                visitor.visitQuantum(instr.operation)
            else:
                raise ParseError(f"Unknown instruction: {instr}")

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
