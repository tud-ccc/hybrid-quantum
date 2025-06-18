//===- Base.td - Quantum dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_BASE
#define QUANTUM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
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
    let hasOperationAttrVerify = 1;
    // let useFoldAPI = kEmitFoldAdaptorFolder;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
        void registerAttributes();
    }];

    let dependentDialects = [
        "arith::ArithDialect"
    ];

    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

// Template for Types
class Quantum_Type<string name, list<Trait> traits = []>
    : TypeDef<Quantum_Dialect, name, traits> {
}

// Template for Ops
class Quantum_Op<string mnemonic, list<Trait> traits = []>
        : Op<Quantum_Dialect, mnemonic, traits>;

// Template for Attributes
class Quantum_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Quantum_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}
      
#endif // QUANTUM_BASE
