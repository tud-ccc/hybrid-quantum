//===- Attributes.td - RVSDG dialect attributes --------*- tablegen -----*-===//
//
// This is the definitions file for the RVSDG dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef RVSDG_ATTRIBUTES
#define RVSDG_ATTRIBUTES

include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.td"

def RVSDG_MatchRuleAttr: RVSDG_Attr<"MatchRule", "matchRule"> {
    let summary = "Attribute for describing mappings for the match op";
    let description = [{
        This attribute is used to describe the mapping of the match op.
        Takes an integer value that is matched to another integer value.
    }];

    let parameters = (ins
        OptionalArrayRefParameter<"int64_t">: $values,
        "uint64_t": $index
    );

    let extraClassDeclaration = [{
        /**
        * Returns true if this mapping rule represents the default case.
        */
        bool isDefault() const { return this->getValues().empty(); }
    }];

    let assemblyFormat = "`<` ( `default` ):($values^)? `->` $index `>`";
}

#endif // RVSDG_ATTRIBUTES
