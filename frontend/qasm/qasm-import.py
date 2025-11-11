#!/usr/bin/env python3
"""
#   Frontend generating QILLR dialect code from QASM2 and QASM3 code.
#
# @author  Washim Neupane (washim.neupane@outlook.com)
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

import argparse
import sys

from mlir.ir import Module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input QASM file")
    parser.add_argument("-o", "--output", help="Output MLIR file")
    parser.add_argument("-r", "--results", action="store_true", help="Emit IR to return the measurement values")
    parser.add_argument("-t", "--target", help="Output MLIR dialect", default="Quantum")
    args = parser.parse_args()

    code: str = open(args.input).read() if args.input else sys.stdin.read()

    if args.target == "QILLR":
        from qasmtoqillr import QASMToMLIR
    else:
        from qasmtoquantum import QASMToMLIR

    module: Module = QASMToMLIR(code, args.results)
    mlir: str = str(module)

    if args.output:
        open(args.output, "w").write(mlir)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
