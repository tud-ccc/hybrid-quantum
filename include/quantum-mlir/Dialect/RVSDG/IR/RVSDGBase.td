//===- Base.td - RVSDG dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the RVSDG dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef RVSDG_BASE
#define RVSDG_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def RVSDG_Dialect : Dialect {
    let name = "rvsdg";
    let cppNamespace = "::mlir::rvsdg";

    let summary = "RVSDG Dialect Base Definition";
    let description = [{
        Regionalized Value State Dependency Graph
    }];

    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 1;
    // let useFoldAPI = kEmitFoldAdaptorFolder;
    let useDefaultAttributePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
        void registerAttributes();
    }];

    let dependentDialects = [
        "arith::ArithDialect"
    ];
}

// Template for Types
class RVSDG_Type<string name, list<Trait> traits = []>
    : TypeDef<RVSDG_Dialect, name, traits> {
}

// Template for Ops
class RVSDG_Op<string mnemonic, list<Trait> traits = []>
    : Op<RVSDG_Dialect, mnemonic, traits>;

// Template for Attributes
class RVSDG_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RVSDG_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}
      
#endif // RVSDG_BASE
