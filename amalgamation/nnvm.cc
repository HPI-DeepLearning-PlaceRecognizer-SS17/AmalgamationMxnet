#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS   0
#else
#define MSHADOW_USE_CBLAS   1
#endif
#endif
#define MSHADOW_USE_CUDA    0
#define MSHADOW_USE_MKL     0
#define MSHADOW_RABIT_PS    0
#define MSHADOW_DIST_PS     0
#define DMLC_LOG_STACK_TRACE 0

#include "mshadow/tensor.h"
#include "mxnet/base.h"
#include "dmlc/json.h"
#include "nnvm/tuple.h"
#include "mxnet/tensor_blob.h"
#include "src/core/graph.cc"
#include "src/core/node.cc"
#include "src/core/op.cc"
#include "src/core/pass.cc"
#include "src/core/symbolic.cc"
#include "src/pass/gradient.cc"
#include "src/pass/infer_shape_type.cc"
#include "src/pass/order_mutation.cc"
#include "src/pass/place_device.cc"
#include "src/pass/plan_memory.cc"
#include "src/pass/saveload_json.cc"
#include "src/c_api/c_api_error.cc"
#include "src/c_api/c_api_graph.cc"
#include "src/c_api/c_api_symbolic.cc"
