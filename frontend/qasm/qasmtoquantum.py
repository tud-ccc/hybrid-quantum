"""
#   Frontend generating Quantum dialect code from QASM2 and QASM3 code.
#
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

from __future__ import annotations

import re
from enum import Enum
from functools import reduce

from mlir._mlir_libs._mlirDialectsQuantum import QuantumQubitType
from mlir._mlir_libs._mlirDialectsQuantum import quantum as quantum_dialect
from mlir._mlir_libs._mlirDialectsRVSDG import ControlType, MatchRuleAttr
from mlir._mlir_libs._mlirDialectsRVSDG import rvsdg as rvsdg_dialect
from mlir.dialects import arith, func, quantum, rvsdg, tensor
from mlir.dialects.builtin import Block, BlockArgumentList, IntegerType, RankedTensorType
from mlir.ir import ArrayAttr, Context, F64Type, InsertionPoint, Location, Module, StringAttr, Type, TypeAttr, Value
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Operation, ParameterExpression, Qubit
from qiskit.circuit import library as lib
from qiskit.circuit.classical.expr import Expr
from qiskit.circuit.controlflow.if_else import IfElseOp
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.qasm2.parse import LEGACY_CUSTOM_INSTRUCTIONS
from qiskit.qasm2.parse import _DefinedGate as QASM2_Gate


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
    def __init__(self, visited: dict[str, quantum.GateOp] | None = None) -> None:
        self.qregs: dict[str, Value | None] = {}
        self.cregs: dict[str, Value | None] = {}
        self.visitedGates: dict[str, quantum.GateOp] = visited if visited is not None else {}

    @classmethod
    def fromList(
        cls,
        qregs: list[QubitSpecifier],
        cregs: list[ClbitSpecifier],
        visited: dict[str, quantum.GateOp] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = {str(q): None for qreg in qregs for q in qreg}
        s.cregs = {str(c): None for creg in cregs for c in creg}
        return s

    @classmethod
    def fromMap(
        cls, qregs: dict[str, Value | None], cregs: dict[str, Value | None], visited: dict[str, quantum.GateOp] | None = None
    ) -> Scope:
        s = cls(visited)
        s.qregs = qregs
        s.cregs = cregs
        return s

    def findAlloc(self, reg: QubitSpecifier) -> Value | None:
        return self.qregs.get(str(reg))

    def setAlloc(self, reg: QubitSpecifier, alloc: Value) -> None:
        self.qregs[str(reg)] = alloc

    def findResult(self, reg: ClbitSpecifier) -> Value | None:
        return self.cregs.get(str(reg))

    def setResult(self, reg: ClbitSpecifier, measurement: Value) -> None:
        self.cregs[str(reg)] = measurement

    def findGate(self, gate: QASM2_Gate) -> quantum.GateOp:
        return self.visitedGates.get(str(gate.name))

    def setGate(self, gate: QASM2_Gate, newGate: quantum.GateOp) -> None:
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
            qubitTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
            alloc: quantum.AllocOp = quantum.AllocOp(qubitTy, loc=self.loc, ip=InsertionPoint(self.block))
            self.scope.setAlloc(reg, alloc.result)

        allocResult: quantum.AllocOp = self.scope.findAlloc(reg)
        return allocResult

    def visitClassicalBit(self, reg: Clbit, *, measurement: Value | None = None) -> Value:
        if measurement is not None:
            self.scope.setResult(reg, measurement)

        val: Value | None = self.scope.findResult(reg)
        if val is None:
            i1Type = IntegerType.get_signless(1)
            zero = arith.ConstantOp(i1Type, 0, ip=InsertionPoint(self.block)).result
            self.scope.setResult(reg, zero)
            return zero
        else:
            return val

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
                    outTy = [arg.type for arg in args]
                    outs: list[Value] = quantum.BarrierOp(outTy, args, ip=InsertionPoint(self.block)).result
                    for q, r in zip(qubits, outs):
                        self.scope.setAlloc(q, r)
            case QASM2_Gate(), _:
                self._visitDefinedGate(instr, qubits, clbits)
            case IfElseOp(), _:
                self._visitIfElse(instr, qubits, clbits)
            case lib.CCXGate(), 3:
                with self.loc:
                    control1: Value = self.visitQuantumBit(qubits[0])
                    control2: Value = self.visitQuantumBit(qubits[1])
                    target: Value = self.visitQuantumBit(qubits[2])
                    ccx: quantum.CCXOp = quantum.CCXOp(control1, control2, target, ip=InsertionPoint(self.block))
                    self.scope.setAlloc(qubits[0], ccx.control1_out)
                    self.scope.setAlloc(qubits[1], ccx.control2_out)
                    self.scope.setAlloc(qubits[2], ccx.target_out)
            case lib.CSwapGate(), 3:
                with self.loc:
                    control: Value = self.visitQuantumBit(qubits[0])
                    lhs: Value = self.visitQuantumBit(qubits[1])
                    rhs: Value = self.visitQuantumBit(qubits[2])
                    cswap: quantum.CSWAPOp = quantum.CSWAPOp(control, lhs, rhs, ip=InsertionPoint(self.block))
                    self.scope.setAlloc(qubits[0], cswap.control_out)
                    self.scope.setAlloc(qubits[1], cswap.lhs_out)
                    self.scope.setAlloc(qubits[2], cswap.rhs_out)
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
                            op: quantum.XOp = quantum.XOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.YGate():
                            op: quantum.YOp = quantum.YOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.ZGate():
                            op: quantum.ZOp = quantum.ZOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.HGate():
                            op: quantum.HOp = quantum.HOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.SGate():
                            op: quantum.SOp = quantum.SOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.SXGate():
                            op: quantum.SXOp = quantum.SXOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.SdgGate():
                            op: quantum.SdgOp = quantum.SdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.TGate():
                            op: quantum.TOp = quantum.TOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.TdgGate():
                            op: quantum.TdgOp = quantum.TdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.RZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RzOp = quantum.RzOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.RXGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RxOp = quantum.RxOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.RYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RyOp = quantum.RyOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.U3Gate():
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.U2Gate():
                            phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U2Op = quantum.U2Op(target, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.U1Gate():
                            lam = self.visitClassic(instr.params[0])
                            op: quantum.U1Op = quantum.U1Op(target, lam, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.UGate():
                            # TODO: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.UGate
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.Reset():
                            outTy = QuantumQubitType.get(self.context, 1)
                            op: quantum.ResetOp = quantum.ResetOp(outTy, target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.Measure():
                            op: quantum.MeasureSingleOp = quantum.MeasureSingleOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                            self.visitClassicalBit(clbits[0], measurement=op.measurement)
                        case lib.IGate():
                            op: quantum.IdOp = quantum.IdOp(target, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
                        case lib.PhaseGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.PhaseOp = quantum.PhaseOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result)
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
                            op: quantum.SWAPOp = quantum.SWAPOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.result_lhs)
                            self.scope.setAlloc(qubits[1], op.result_rhs)
                        case lib.CZGate():
                            op: quantum.CZOp = quantum.CZOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.control_out)
                            self.scope.setAlloc(qubits[1], op.target_out)
                        case lib.CXGate():
                            op: quantum.CNOTOp = quantum.CNOTOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.control_out)
                            self.scope.setAlloc(qubits[1], op.target_out)
                        case lib.CRYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRyOp = quantum.CRyOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.control_out)
                            self.scope.setAlloc(qubits[1], op.target_out)
                        case lib.CRZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRzOp = quantum.CRzOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.control_out)
                            self.scope.setAlloc(qubits[1], op.target_out)
                        case lib.CU1Gate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CU1Op = quantum.CU1Op(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setAlloc(qubits[0], op.control_out)
                            self.scope.setAlloc(qubits[1], op.target_out)

    def _visitDefinedGate(self, instr: QASM2_Gate, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        if instr.definition is not None:
            if self.scope.findGate(instr) is None:
                # Construct quantum.GateOp for defined custom gate
                # Insert into module body and recursively visit gate body
                inputTypes: list[QuantumQubitType] = [QuantumQubitType.get(self.context, 1)] * instr.num_qubits
                gty: func.FunctionType = func.FunctionType.get(inputs=inputTypes, results=inputTypes, context=self.context)
                gate: quantum.GateOp = quantum.GateOp(
                    StringAttr.get(str(instr.name)),
                    TypeAttr.get(gty),
                    loc=self.loc,
                    ip=InsertionPoint.at_block_begin(self.module.body),
                )
                gateBody: Block = gate.body.blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))

                self.scope.setGate(instr, gate)
                circuit: QuantumCircuit = instr.definition
                gateQregs = {str(q): a for q, a in zip(circuit.qubits, gate.body.blocks[0].arguments)}
                innerGateScope: Scope = Scope.fromMap(gateQregs, {}, self.scope.visitedGates)
                visitor: QASMToMLIRVisitor = QASMToMLIRVisitor.fromParent(self, block=gateBody, scope=innerGateScope)
                visitor.visitCircuit(circuit)
                quantum.ReturnOp([visitor.visitQuantumBit(q) for q in gateQregs], loc=self.loc, ip=InsertionPoint(gateBody))
            # Construct quantum.CallOp for defined custom gate
            # TODO: qpu.circuit instead of GateOp
            callee: StringAttr = instr.name
            operands: list[Value] = [self.visitQuantumBit(q) for q in qubits]
            outTys: list[Type] = [o.type for o in operands]
            op: quantum.GateCallOp = quantum.GateCallOp(outTys, callee, operands, loc=self.loc, ip=InsertionPoint(self.block))
            for inq, outq in zip(qubits, op.results):
                self.scope.setAlloc(inq, outq)
        else:
            ParseError(f"Expected gate with definition, got: {instr}")

    def _visitIfElse(self, instr: IfElseOp, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        with self.loc:
            condition: Value = self._visitIfElseCondition(instr.condition, qubits, clbits)
            true_body, false_body = instr.params
            hasElse: bool = false_body is not None

            matchTy: ControlType = ControlType.get(self.context, 2)
            mapTrue: MatchRuleAttr = MatchRuleAttr.get(self.context, [1], 0)
            mapFalse: MatchRuleAttr = MatchRuleAttr.get(self.context, [0], 1)
            mappings: ArrayAttr = ArrayAttr.get([mapTrue, mapFalse])
            predicate: rvsdg.MatchOp = rvsdg.MatchOp(matchTy, condition, mappings, ip=InsertionPoint(self.block))

            ifTy = [QuantumQubitType.get(self.context, 1)] * len(qubits)
            inputs = [self.visitQuantumBit(q) for q in qubits]
            ifOp: rvsdg.GammaNode = rvsdg.GammaNode(ifTy, predicate, inputs, 2, ip=InsertionPoint(self.block))
            inputTypes = [inp.type for inp in inputs]

            thenBlock: Block = ifOp.regions[0].blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))
            thenScope: Scope = Scope.fromMap(
                qregs={str(q): v for q, v in zip(qubits, iter_block_args(thenBlock.arguments))},
                cregs=self.scope.cregs,
                visited=self.scope.visitedGates,
            )
            thenVisitor = QASMToMLIRVisitor.fromParent(self, block=thenBlock, scope=thenScope)
            thenVisitor.visitCircuit(true_body)
            rvsdg.YieldOp([thenVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(thenBlock))

            elseBlock: Block = ifOp.regions[1].blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))
            elseScope: Scope = Scope.fromMap(
                qregs={str(q): v for q, v in zip(qubits, iter_block_args(elseBlock.arguments))},
                cregs=self.scope.cregs,
                visited=self.scope.visitedGates,
            )
            elseVisitor = QASMToMLIRVisitor.fromParent(self, block=elseBlock, scope=elseScope)
            if hasElse:
                # Visit the else circuit only if it exists
                elseVisitor.visitCircuit(false_body)

            rvsdg.YieldOp([elseVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(elseBlock))

            # Update qubits with returned values
            for inq, outv in zip(qubits, ifOp.outputs):
                self.scope.setAlloc(inq, outv)

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
                            measurement: Value = self.visitClassicalBit(bitOrRegister)
                            axiomval: Value = arith.ConstantOp(i1Type, axiom, ip=InsertionPoint(self.block)).result
                            return arith.CmpIOp(
                                arith.CmpIPredicate.eq, measurement, axiomval, ip=InsertionPoint(self.block)
                            ).result
                        case ClassicalRegister():
                            clvals: list[tuple[int, Value]] = [
                                (bit_at(axiom, i), clbit) for i, clbit in enumerate(clbits[0]._register)
                            ]
                            cmpis: list[Value] = []
                            for b, clbit in clvals:
                                measurement: Value = self.visitClassicalBit(clbit)
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


def QASMToMLIR(code: str, emitResults: bool) -> Module:
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
    quantum_dialect.register_dialect(context)
    rvsdg_dialect.register_dialect(context)

    with context:
        location: Location = Location.unknown()
        module: Module = Module.create(location)

        scope: Scope = Scope.fromList(circuit.qregs, circuit.cregs)

        qasm_main: func.FuncOp = func.FuncOp("qasm_main", ([], []), visibility="public", loc=location)
        qasm_main.add_entry_block()

        visitor: QASMToMLIRVisitor = QASMToMLIRVisitor(compat, context, module, location, qasm_main.entry_block, scope)
        visitor.visitCircuit(circuit)

        for qubit in circuit.qubits:
            qubitValue: Value = visitor.visitQuantumBit(qubit)
            quantum.DeallocateOp(qubitValue, loc=visitor.loc, ip=InsertionPoint(visitor.block))

        if not emitResults:
            func.ReturnOp([], loc=location, ip=InsertionPoint(qasm_main.entry_block))
        else:
            # Create a new main function with the correct type and move the body to it
            m: list[Value] = [r for _, r in scope.cregs.items() if r is not None]
            resType: RankedTensorType = RankedTensorType.get([len(m)], IntegerType.get_signless(1), loc=location)
            qasm_main.attributes["function_type"] = TypeAttr.get(func.FunctionType.get([], [resType]))
            # Merge all measurements into a tensor and return it
            res: Value = tensor.FromElementsOp(resType, m, loc=location, ip=InsertionPoint(qasm_main.entry_block)).result
            func.ReturnOp([res], loc=location, ip=InsertionPoint(qasm_main.entry_block))

    module.body.append(qasm_main)
    return module


def bit_at(n: int, i: int) -> int:
    """Return the value (0 or 1) of bit *i* of n."""
    return (n >> i) & 1


def iter_block_args(bal: BlockArgumentList):
    for i in range(len(bal)):
        yield bal[i]
