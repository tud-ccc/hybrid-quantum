//===- Base.td - QILLR dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the QILLR dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef QILLR_BASE
#define QILLR_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def QILLR_Dialect : Dialect {
    let name = "qillr";
    let cppNamespace = "::mlir::qillr";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;
    //let useFoldAPI = kEmitFoldAdaptorFolder;

    let dependentDialects = [
        "mlir::quantum::QuantumDialect",
        "mlir::tensor::TensorDialect",
        "mlir::scf::SCFDialect"
    ];

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
// class QILLR_Attr<string name, list<Trait> traits = []>
//         : AttrDef<QILLR_Dialect, name, traits>;
// Template for ops.
class QILLR_Op<string mnemonic, list<Trait> traits = []>
        : Op<QILLR_Dialect, mnemonic, traits>;
        
// Template for types.
class QILLR_Type<string name, list<Trait> traits = []>
        : TypeDef<QILLR_Dialect, name, traits>;

class Gate_Op<string mnemonic, list<Trait> traits = []>
        : QILLR_Op<mnemonic, traits>;

class Memory_Op<string mnemonic, list<Trait> traits = []> :
        QILLR_Op<mnemonic, traits>;

#endif
