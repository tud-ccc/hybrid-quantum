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
include "mlir/Interfaces/SideEffectInterfaces.td"


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
    public:
        static StringRef getContainerModuleAttrName() {
            return "quantum.container_module";
        }
        static StringRef getKernelFuncAttrName() { return "quantum.kernel"; }
        static bool isKernel(Operation *op);
        static unsigned getNumWorkgroupDimensions() { return 1; }
        // static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }
        // static AddressSpace getPrivateAddressSpace() { return AddressSpace::Private; }
    }];
    // let dependentDialects = ["arith::ArithDialect"];
    let useDefaultAttributePrinterParser = 1;
    // let useDefaultTypePrinterParser = 1;
}

class Quantum_Op<string mnemonic, list<Trait> traits = []>
        : Op<Quantum_Dialect, mnemonic, traits>;

class Quantum_Type<string name, list<Trait> traits = []>
    : TypeDef<Quantum_Dialect, name, traits> {
}


class Gate_Op<string mnemonic, list<Trait> traits = []>
        : Quantum_Op<mnemonic, traits>;

class Memory_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits>;

class Observable_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits # [Pure]>;

class Measurement_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits>; 
        

//===----------------------------------------------------------------------===//
// Quantum dialect traits.
//===----------------------------------------------------------------------===//
def Unitary : NativeOpTrait<"Unitary">;
def Hermitian : NativeOpTrait<"Hermitian">;
def Kernel : NativeOpTrait<"Kernel">;
def QuantumMemory : Resource<"QuantumMemory">;
        
#endif
