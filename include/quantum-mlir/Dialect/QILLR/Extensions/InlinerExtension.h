//===- InlinerExtension.h - QILLR Inliner Extension -------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension for the QILLR dialect that implements the
// interfaces necessary to support inlining of gates.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;

namespace qillr {
/// Register the extension used to support inlining the QILLR dialect.
void registerInlinerExtension(DialectRegistry &registry);
} // namespace qillr

} // namespace mlir
