//===- Base.td - QIR dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_BASE
#define QIR_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def QIR_Dialect : Dialect {
    let name = "qir";
    let cppNamespace = "::mlir::qir";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;
    //let useFoldAPI = kEmitFoldAdaptorFolder;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
// class QIR_Attr<string name, list<Trait> traits = []>
//         : AttrDef<QIR_Dialect, name, traits>;
// Template for ops.
class QIR_Op<string mnemonic, list<Trait> traits = []>
        : Op<QIR_Dialect, mnemonic, traits>;
        
// Template for types.
class QIR_Type<string name, list<Trait> traits = []>
        : TypeDef<QIR_Dialect, name, traits>;

class Gate_Op<string mnemonic, list<Trait> traits = []>
        : QIR_Op<mnemonic, traits>;

class Memory_Op<string mnemonic, list<Trait> traits = []> :
        QIR_Op<mnemonic, traits>;

#endif
