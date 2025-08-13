//===- Base.td - QQT dialect base --------------------------*- tablegen -*-===//
//
// This is the definitions file for the QQT dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef QQT_BASE
#define QQT_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def QQT_Dialect : Dialect {
    let name = "qqt";
    let cppNamespace = "::mlir::qqt";

    let summary = "QILLR Quantum Transform Dialect";
    let description = [{
        The dialect enables the QILLR to Quantum transformation in the presence
        of control-flow operations without violating the linear type semantics
        of the quantum types.
    }];

    //let hasOperationAttrVerify = 1;    
    //let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;

    let dependentDialects = [
        "::mlir::quantum::QuantumDialect",
        "::mlir::qillr::QILLRDialect",
        "::mlir::rvsdg::RVSDGDialect"
    ];

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
        void registerAttributes();
    }];
}

// Template for Types
class QQT_Type<string name, list<Trait> traits = []>
    : TypeDef<QQT_Dialect, name, traits> {
}

// Template for Ops
class QQT_Op<string mnemonic, list<Trait> traits = []>
        : Op<QQT_Dialect, mnemonic, traits>;

// Template for Attributes
class QQT_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QQT_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // QQT_BASE
