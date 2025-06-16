//===- Types.td - QIR dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_TYPES
#define QIR_TYPES

include "quantum-mlir/Dialect/QIR/IR/QIRAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def QIR_QubitType : QIR_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "qubit type";
  code builderCall = "::mlir::qir::QubitType::get($_ctxt)";
}

def QIR_ResultType : QIR_Type<"Result"> {
    let mnemonic = "result";
    let summary = "A quantum measurement result.";
}

def QIR_MeasurementType : 1DTensorOf<[I1]>;

def QIR_DeviceType : QIR_Type<"Device"> {
  let mnemonic = "device";
  let description = "Quantum device type that holds information about the device size and coupling graph.";
  let parameters = (ins "int64_t":$num_qubits, "ArrayAttr":$coupling_graph);
  let assemblyFormat = "`<` $num_qubits `,` $coupling_graph `>`";
}

#endif
