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
from functools import reduce

from mlir._mlir_libs._mlirDialectsQIR import QubitType
from mlir._mlir_libs._mlirDialectsQIR import qir as qirdialect
from mlir.dialects import arith, func, qir, scf
from mlir.dialects.builtin import Block, IntegerType
from mlir.ir import Context, F64Type, InsertionPoint, Location, Module, StringAttr, TypeAttr, Value
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Operation, ParameterExpression, Qubit
from qiskit.circuit import library as lib
from qiskit.circuit.classical.expr import Expr
from qiskit.circuit.controlflow.if_else import IfElseOp
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
type QubitSpecifier = Qubit | QuantumRegister
# int,
# slice,
# Sequence[Union[Qubit, int]],

# Types that can be coerced to a valid Clbit specifier in a circuit.
type ClbitSpecifier = Clbit | ClassicalRegister
# int,
# slice,
# Sequence[Union[Clbit, int]],


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
        cregs: dict[str, qir.AllocResultOp],
        visited: dict[str, qir.GateOp] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = qregs
        s.cregs = cregs
        return s

    def findAlloc(self, reg: QubitSpecifier) -> qir.AllocOp:
        return self.qregs.get(str(reg))

    def setAlloc(self, reg: QubitSpecifier, alloc: qir.AllocOp) -> None:
        self.qregs[str(reg)] = alloc

    def findResult(self, reg: ClbitSpecifier) -> qir.AllocResultOp:
        return self.qregs.get(str(reg))

    def setResult(self, reg: ClbitSpecifier, ralloc: qir.AllocResultOp) -> None:
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

    def visitQuantumBit(self, reg: Qubit) -> Value:
        if self.scope.findAlloc(reg) is None:
            alloc: qir.AllocOp = qir.AllocOp(loc=self.loc, ip=InsertionPoint(self.block))
            self.scope.setAlloc(reg, alloc.result)
            return alloc.result

        return self.scope.findAlloc(reg)

    def visitClassicalBit(self, reg: Clbit) -> Value:
        if self.scope.findResult(reg) is None:
            ralloc: qir.AllocResultOp = qir.AllocResultOp(loc=self.loc, ip=InsertionPoint(self.block))
            self.scope.setResult(reg, ralloc.result)
            return ralloc.result

        return self.scope.findResult(reg)

    def visitClassic(self, expr: Expr) -> Value:
        if isinstance(expr, ParameterExpression):
            raise NotImplementedError("Parameter Expression")
        elif isinstance(expr, float):
            return arith.ConstantOp(F64Type.get(self.context), expr, ip=InsertionPoint(self.block)).result
        else:
            raise NotImplementedError(f"Classic expressions are not supported for {expr}")

    # Operation encapsulates virtual instructions that must
    # be synthesized to physical instructions
    def visitQuantum(self, instr: Operation) -> None:
        raise NotImplementedError(f"Virtual quantum expressions are not supported for {instr}")

    # Instruction represents physical quantum instructions
    def visitInstruction(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        match instr, len(qubits):
            case lib.Barrier(), _:
                with self.loc:
                    args = [self.visitQuantumBit(q) for q in qubits]
                    qir.BarrierOp(args, ip=InsertionPoint(self.block))
            case QASM2_Gate(), _:
                self._visitDefinedGate(instr, qubits, clbits)
            case IfElseOp(), _:
                self._visitIfElse(instr, qubits, clbits)
            case lib.CCXGate(), 3:
                with self.loc:
                    control1: Value = self.visitQuantumBit(qubits[0])
                    control2: Value = self.visitQuantumBit(qubits[1])
                    target: Value = self.visitQuantumBit(qubits[2])
                    qir.CCXOp(control1, control2, target, ip=InsertionPoint(self.block))
            case lib.CSwapGate(), 3:
                with self.loc:
                    control: Value = self.visitQuantumBit(qubits[0])
                    lhs: Value = self.visitQuantumBit(qubits[1])
                    rhs: Value = self.visitQuantumBit(qubits[2])
                    qir.CSwapOp(control, lhs, rhs, ip=InsertionPoint(self.block))
            case _, 1:
                self._visitUnaryGates(instr, qubits, clbits)
            case _, 2:
                self._visitBinaryGates(instr, qubits, clbits)
            case _, _:
                raise NotImplementedError(f"{instr} of type {type(instr)}")

    def _visitUnaryGates(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        assert len(qubits) == 1, f"Require unary gate, got: {instr}"
        match qubits[0]:
            case QuantumRegister():
                raise NotImplementedError(f"Visiting instruction {instr} with QuantumRegister {qubits} and {clbits}")
            case Qubit():
                with self.loc:
                    target: Value = self.visitQuantumBit(qubits[0])
                    match instr:
                        case lib.XGate():
                            qir.XOp(target, ip=InsertionPoint(self.block))
                        case lib.YGate():
                            qir.YOp(target, ip=InsertionPoint(self.block))
                        case lib.ZGate():
                            qir.ZOp(target, ip=InsertionPoint(self.block))
                        case lib.HGate():
                            qir.HOp(target, ip=InsertionPoint(self.block))
                        case lib.SGate():
                            qir.SOp(target, ip=InsertionPoint(self.block))
                        case lib.SXGate():
                            qir.SXOp(target, ip=InsertionPoint(self.block))
                        case lib.SdgGate():
                            qir.SdgOp(target, ip=InsertionPoint(self.block))
                        case lib.TGate():
                            qir.TOp(target, ip=InsertionPoint(self.block))
                        case lib.TdgGate():
                            qir.TdgOp(target, ip=InsertionPoint(self.block))
                        case lib.RZGate():
                            angle = self.visitClassic(instr.params[0])
                            qir.RzOp(target, angle, ip=InsertionPoint(self.block))
                        case lib.RXGate():
                            angle = self.visitClassic(instr.params[0])
                            qir.RxOp(target, angle, ip=InsertionPoint(self.block))
                        case lib.RYGate():
                            angle = self.visitClassic(instr.params[0])
                            qir.RyOp(target, angle, ip=InsertionPoint(self.block))
                        case lib.U3Gate():
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            qir.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                        case lib.U2Gate():
                            phi, lam = [self.visitClassic(param) for param in instr.params]
                            qir.U2Op(target, phi, lam, ip=InsertionPoint(self.block))
                        case lib.U1Gate():
                            lam = self.visitClassic(instr.params[0])
                            qir.U1Op(target, lam, ip=InsertionPoint(self.block))
                        case lib.Reset():
                            qir.ResetOp(target, ip=InsertionPoint(self.block))
                        case lib.Measure():
                            bit: Value = self.visitClassicalBit(clbits[0])
                            measureOp: qir.MeasureOp = qir.MeasureOp(target, bit, ip=InsertionPoint(self.block))
                            qir.ReadMeasurementOp(measureOp.result, ip=InsertionPoint(self.block))
                        case lib.IGate():
                            qir.IdOp(target, ip=InsertionPoint(self.block))
                        case _:
                            raise NotImplementedError(f"Unary gate {instr}")

    def _visitBinaryGates(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        assert len(qubits) == 2, f"Require binary gate, got: {instr}"
        match qubits[0], qubits[1]:
            case QuantumRegister(), QuantumRegister():
                raise NotImplementedError(f"Visiting instruction {instr} with QuantumRegister {qubits} and {clbits}")
            case Qubit(), Qubit():
                with self.loc:
                    lhs: Value = self.visitQuantumBit(qubits[0])
                    rhs: Value = self.visitQuantumBit(qubits[1])
                    match instr:
                        case lib.SwapGate():
                            qir.SwapOp(lhs, rhs, ip=InsertionPoint(self.block))
                        case lib.CZGate():
                            qir.CZOp(lhs, rhs, ip=InsertionPoint(self.block))
                        case lib.CXGate():
                            qir.CNOTOp(lhs, rhs, ip=InsertionPoint(self.block))
                        case lib.CRYGate():
                            angle = self.visitClassic(instr.params[0])
                            qir.CRyOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                        case lib.CRZGate():
                            angle = self.visitClassic(instr.params[0])
                            qir.CRzOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                        case lib.CU1Gate():
                            angle = self.visitClassic(instr.params[0])
                            qir.CU1Op(lhs, rhs, angle, ip=InsertionPoint(self.block))

    def _visitDefinedGate(self, instr: QASM2_Gate, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        if instr.definition is not None:
            if self.scope.findGate(instr) is None:
                # Construct qir.GateOp for dDefined custom gate
                # Insert into module body and recursively visit gate body
                inputs: list[QubitType] = [QubitType.get(self.context) for _ in range(instr.num_qubits)]
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
            callee: StringAttr = instr.name
            operands: list[Value] = [self.visitQuantumBit(q) for q in qubits]
            qir.GateCallOp(callee, operands, loc=self.loc, ip=InsertionPoint(self.block))
        else:
            ParseError(f"Expected gate with definition, got: {instr}")

    def _visitIfElse(self, instr: IfElseOp, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        with self.loc:
            condition: Value = self._visitIfElseCondition(instr.condition, qubits, clbits)
            true_body, false_body = instr.params
            hasElse: bool = false_body is not None

            ifOp: scf.IfOp = scf.IfOp(condition, hasElse=hasElse, ip=InsertionPoint(self.block))

            thenVisitor = QASMToMLIRVisitor.fromParent(self, block=ifOp.then_block)
            thenVisitor.visitCircuit(true_body)
            if hasElse:
                elseVisitor = QASMToMLIRVisitor.fromParent(self, block=ifOp.else_block)
                elseVisitor.visitCircuit(false_body)

    def _visitIfElseCondition(
        self,
        condition: Expr | tuple[ClassicalRegister, int] | tuple[Clbit, int],
        qubits: list[QubitSpecifier],
        clbits: list[ClbitSpecifier],
    ) -> Value:
        match condition:
            case Expr():
                raise NotImplementedError(f" IfElseOp with condition of type {type(condition)}")
            case (bitOrRegister, axiom):  # tuple[ClassicalRegister, int] | tuple[Clbit, int]
                with self.loc:
                    i1Type = IntegerType.get_signless(1)
                    match bitOrRegister:
                        case Clbit():
                            clval: Value = self.visitClassicalBit(bitOrRegister)
                            measurement: Value = qir.ReadMeasurementOp(clval, ip=InsertionPoint(self.block)).result
                            axiomval: Value = arith.ConstantOp(i1Type, axiom, ip=InsertionPoint(self.block)).result
                            return arith.CmpIOp(
                                arith.CmpIPredicate.eq, measurement, axiomval, ip=InsertionPoint(self.block)
                            ).result
                        case ClassicalRegister():
                            clvals: list[tuple[int, Value]] = [
                                (bit_at(axiom, i), self.visitClassicalBit(clbit)) for i, clbit in enumerate(clbits[0]._register)
                            ]
                            cmpis: list[Value] = []
                            for b, clval in clvals:
                                measurement: Value = qir.ReadMeasurementOp(clval, ip=InsertionPoint(self.block)).result
                                axiomval: Value = arith.ConstantOp(i1Type, b, ip=InsertionPoint(self.block)).result
                                cmpis.append(
                                    arith.CmpIOp(
                                        arith.CmpIPredicate.eq, measurement, axiomval, ip=InsertionPoint(self.block)
                                    ).result
                                )
                            return reduce(
                                lambda cmps, cmp: arith.AndIOp(cmps, cmp, ip=InsertionPoint(self.block)).result,
                                cmpis[1:],  # rest of the list
                                cmpis[0],  # initial value
                            )
                        case _:
                            raise NotImplementedError(f"condition of type {type(condition)}")


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
    qirdialect.register_dialect(context)

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


def bit_at(n: int, i: int) -> int:
    """Return the value (0 or 1) of bit *i* of n."""
    return (n >> i) & 1


if __name__ == "__main__":
    main()
