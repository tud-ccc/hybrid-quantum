//===- Types.td - RVSDG dialect types ------------------*- tablegen -----*-===//
//
// This is the definitions file for the RVSDG dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef RVSDG_TYPES
#define RVSDG_TYPES

include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def RVSDG_Ctrl : RVSDG_Type<"Control"> {
    let mnemonic = "ctrl";
    let summary = "Type used as predicate for gamma and theta nodes";
    let description = [{
        Type used as predicate for gamma and theta nodes. Is generated
        by the match op. Has a fixed number of possible values.
    }];

    let parameters = (ins
        "uint64_t":$numOptions
    );

    let builders = [
        TypeBuilder<(ins "uint64_t":$numOptions)>
    ];

    let assemblyFormat = "`<` $numOptions `>`";
}

class RVSDG_Ctrl<int N>: 
ConfinedType<RVSDG_Ctrl, 
    [CPred<"$_self.cast<::mlir::rvsdg::ControlType>().getNumOptions() == " # N # "">],
    "RVSDG control type with " # N # " options"
>,
BuildableType<"::mlir::rvsdg::ControlType::get($_builder.getContext(), " # N # ")"> 
;

#endif
