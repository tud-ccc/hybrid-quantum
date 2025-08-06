// RUN: quantum-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-DAG: quantum
// CHECK-DAG: qillr
// CHECK-DAG: qqt
// CHECK-DAG: rvsdg
