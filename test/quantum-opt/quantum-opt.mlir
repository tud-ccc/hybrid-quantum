// RUN: quantum-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-DAG: quantum
// CHECK-DAG: QILLR
// CHECK-DAG: QQT
// CHECK-DAG: RVSDG
