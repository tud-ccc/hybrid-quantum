//===- Base.td - Quantum dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_BASE
#define QUANTUM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def Quantum_Dialect : Dialect {
    let name = "quantum";
    let cppNamespace = "::mlir::quantum";

    let summary = "Quantum Dialect Base Definition";
    let description = [{
        Contains all base def. 
    }];

    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 0;
    // let useFoldAPI = kEmitFoldAdaptorFolder;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];

    let dependentDialects = [
        "arith::ArithDialect"
    ];

    let useDefaultAttributePrinterParser = 0;
    // let useDefaultTypePrinterParser = 1;
}

// Template for Types
class Quantum_Type<string name, list<Trait> traits = []>
    : TypeDef<Quantum_Dialect, name, traits> {
}

// Template for Ops
class Quantum_Op<string mnemonic, list<Trait> traits = []>
        : Op<Quantum_Dialect, mnemonic, traits>;
      
#endif // QUANTUM_BASE
