//===- Ops.td - RVSDG dialect ops ----------------------*- tablegen -----*-===//
//
// This is the definitions file for the RVSDG dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef RVSDG_OPS
#define RVSDG_OPS

include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// RVSDG OP templates
//===----------------------------------------------------------------------===//

class RVSDG_StructuralNode<string mnemonic, list<Trait> traits = []> : 
    RVSDG_Op<mnemonic, traits # [IsolatedFromAbove]>;

//===----------------------------------------------------------------------===//
// RVSDG dialect traits.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RVSDG Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Control operations
//===----------------------------------------------------------------------===//

def RVSDG_ConstantCtrl: RVSDG_Op<"constantCtrl", [Pure]> {
    let summary = "Materialize constant control value from integer constant";
    let description = [{
        Creates a constant control value from a positive integer constant.
    }];

    let arguments = (ins
        UI64Attr:$value
    );

    let results = (outs
        RVSDG_Ctrl:$output
    );

    let assemblyFormat = "attr-dict $value `:` type($output)";
}

def RVSDG_MatchOp: RVSDG_Op<"match", [Pure]> {
    let summary = "Match operator. Lookup table that converts integers into control values";
    let description = [{
        Match operator. Lookup table that converts integers into control values.
        The first input is the integer to be matched. The second input is the
        default control value. The remaining inputs are pairs of integers and
        control values. The first integer of each pair is matched against the
        first input. If the match succeeds, the corresponding control value is
        returned. If no match is found, the default control value is returned.
    }];

    let arguments = (ins
        AnyInteger:$input,
        TypedArrayAttrBase<RVSDG_MatchRuleAttr, "Array of mapping rules">:$mapping
    );

    let results = (outs
        RVSDG_Ctrl:$output
    );

    let assemblyFormat = "attr-dict `(` $input `:` type($input) `)` $mapping `->` type($output)";
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Gamma Node (if, switch)
//===----------------------------------------------------------------------===//
def RVSDG_GammaNode: RVSDG_StructuralNode<"gammaNode"> {
    let summary = "RVSDG gamma node which models conditional execution";
    let description = [{
        RVSDG gamma node. Models conditional execution such as if and switch statements.
        The node takes a predicate as its first input. The value of the predicate is used
        as the index of the region to be executed. Any other inputs are passed as arguments
        to all regions of the gamma node. The results of all regions are mapped to the
        outputs of the gamma node.
    }];
    let arguments = (ins
        RVSDG_Ctrl:$predicate,
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$regions
    );

    let assemblyFormat = "attr-dict `(` $predicate `:` type($predicate) `)` custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegions>($regions) `->` type(results)";
    let hasVerifier = 1;
}

def RVSDG_YieldOp: RVSDG_Op<"yield", [
        HasParent<"mlir::rvsdg::GammaNode">,
        Terminator
    ]> {
    let summary = "Gamma node results";
    let description = [{
        Gamma node terminator. Maps operands to gamma node
        output.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";
    let hasVerifier = 1;
}

#endif // RVSDG_OPS
