#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "llvm/ADT/StringExtras.h"

#include <doctest/doctest.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using namespace mlir::quantum;

// clang-format off

TEST_CASE("QuantumIf::Builder creates a valid QuantumIf") {
    MLIRContext context;
    context.loadDialect<mlir::quantum::QuantumDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Create a boolean type (i1)
    auto i1Type = builder.getI1Type();
    // Create a constant with value `true` (1)
    auto trueConst = builder.create<mlir::arith::ConstantOp>(
        loc,
        mlir::IntegerAttr::get(i1Type, 1));
    auto qtype = QubitType::get(&context, 1);

    SUBCASE("capture nothing") {
        auto op = builder.create<IfOp>(loc, trueConst, ValueRange{});
        
        REQUIRE(op.getNumConditionVars() == 1);
        auto regionCapturedArgs = op.getRegionCapturedArgs();
        REQUIRE(regionCapturedArgs.size() == 0);
        REQUIRE(op.getNumRegionCapturedArgs() == 0);
        REQUIRE(op->getNumResults() == 0);
    }

    SUBCASE("capture one value") {
        auto q = builder.create<AllocOp>(
            loc,
            qtype).getResult();
        auto op = builder.create<IfOp>(loc, trueConst, q);
        
        auto regionCapturedArgs = op.getRegionCapturedArgs();
        REQUIRE(regionCapturedArgs.size() == 1);
        REQUIRE(op.getNumRegionCapturedArgs() == 1);
        REQUIRE(op->getNumResults() == 1);
    }
}

TEST_CASE("QuantumIf parsed") {
    MLIRContext context;
    DialectRegistry registry;
    registry.insert<mlir::quantum::QuantumDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    const char *source = R"mlir(
        module {
            %b = index.bool.constant true
            %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
            %out = quantum.if %b ins(%arg0 = %q) -> (!quantum.qubit<1>) {
                quantum.yield %arg0 : !quantum.qubit<1>
            } else {
                quantum.yield %arg0 : !quantum.qubit<1>
            }
        }
    )mlir";

  OwningOpRef<Operation *> module = parseSourceString<ModuleOp>(source, &context);
}