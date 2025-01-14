#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>

module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<1x3x60x60xf32>) -> tensor<180x60xf32> {
    %cst = arith.constant dense_resource<torch_tensor_60_60_torch.float32> : tensor<60x60xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_60_torch.float32> : tensor<60xf32>

    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<1x3x60x60xf32> into tensor<180x60xf32>
    %0 = tensor.empty() : tensor<60x60xf32>
    
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<60x60xf32>) outs(%0 : tensor<60x60xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60x60xf32>

    %2 = tensor.empty() : tensor<180x60xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<180x60xf32>) -> tensor<180x60xf32>
    %4 = linalg.matmul ins(%collapsed, %1 : tensor<180x60xf32>, tensor<60x60xf32>) outs(%3 : tensor<180x60xf32>) -> tensor<180x60xf32>
    
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_2 : tensor<180x60xf32>, tensor<60xf32>) outs(%2 : tensor<180x60xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %27 = arith.addf %in, %in_15 : f32
      linalg.yield %27 : f32
    } -> tensor<180x60xf32>
    return %5 : tensor<180x60xf32>
  }
}
