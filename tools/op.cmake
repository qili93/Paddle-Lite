set(NET "googlenet" CACHE STRING "select net type")
set_property(CACHE NET PROPERTY STRINGS "defult" "googlenet" "mobilenet" "yolo" "squeezenet")

if (NET EQUAL "googlenet")
  set(CONCAT_OP ON)
  set(CONV_OP ON)
  set(LRN_OP ON)
  set(MUL_OP ON)
  set(ELEMENTWISEADD_OP ON)
  set(FUSION_FC_OP ON)
  set(POOL_OP ON)
  set(RELU_OP ON)
  set(FUSION_CONVADD_OP ON)
  set(FUSION_CONVADD_RELU_OP ON)
elseif (NET EQUAL "mobilenet")
  set(CONV_OP ON)
  set(ELEMENTWISEADD_OP ON)
  set(RELU_OP ON)
  set(SOFTMAX_OP ON)
  set(SOFTMAX_OP ON)
  set(DEPTHWISECONV_OP ON)
  set(BATCHNORM_OP ON)
  set(POOL_OP ON)
  set(RESHAPE_OP ON)
  set(FUSION_CONVADDBNRELU_OP)
elseif (NET EQUAL "yolo")
  set(BATCHNORM_OP ON)
  set(CONV_OP ON)
  set(RELU_OP ON)
  set(ELEMENTWISEADD_OP ON)
elseif (NET EQUAL "squeezenet")
  set(CONCAT_OP ON)
  set(CONV_OP ON)
  set(RELU_OP ON)
  set(ELEMENTWISEADD_OP ON)
  set(POOL_OP ON)
  set(RESHAPE_OP ON)
  set(SOFTMAX_OP ON)
elseif (NET EQUAL "resnet")
  set(CONV_OP ON)
  set(BATCHNORM_OP ON)
  set(ELEMENTWISEADD_OP ON)
  set(SOFTMAX_OP ON)
  set(MUL_OP ON)
  set(POOL_OP ON)
  set(RELU_OP ON)
else ()
  set(BATCHNORM_OP ON)
  set(BOXCODER_OP ON)
  set(CONCAT_OP ON)
  set(CONV_OP ON)
  set(DEPTHWISECONV_OP ON)
  set(ELEMENTWISEADD_OP ON)
  set(FUSION_CONVADD_OP ON)
  set(CONVADDRELU_OP ON)
  set(FUSION_FC_OP ON)
  set(LRN_OP ON)
  set(MUL_OP ON)
  set(MULTICLASSNMS_OP ON)
  set(POOL_OP ON)
  set(PRIORBOX_OP ON)
  set(RELU_OP ON)
  set(RESHAPE_OP ON)
  set(SIGMOID_OP ON)
  set(SOFTMAX_OP ON)
  set(TRANSPOSE_OP ON)
  set(FUSION_CONVADD_RELU_OP ON)
  set(FUSION_CONVADDBNRELU_OP ON)

  # option(BATCHNORM_OP "" ON)
  # option(BOXCODER_OP "" ON)
  # option(CONCAT_OP "" ON)
  # option(CONV_OP "" ON)
  # option(DEPTHWISECONV_OP "" ON)
  # option(ELEMENTWISEADD_OP "" ON)
  # option(FUSION_CONVADD_OP "" ON)
  # option(CONVADDRELU_OP "" ON)
  # option(FUSION_FC_OP "" ON)
  # option(LRN_OP "" ON)
  # option(MUL_OP "" ON)
  # option(MULTICLASSNMS_OP "" ON)
  # option(POOL_OP "" ON)
  # option(PRIORBOX_OP "" ON)
  # option(RELU_OP "" ON)
  # option(RESHAPE_OP "" ON)
  # option(SIGMOID_OP "" ON)
  # option(SOFTMAX_OP "" ON)
  # option(TRANSPOSE_OP "" ON)
  # option(FUSION_CONVADD_RELU_OP "" ON)
endif ()

if (BATCHNORM_OP)
  add_definitions(-DBATCHNORM_OP)
endif()
if (BOXCODER_OP)
  add_definitions(-DBOXCODER_OP)
endif()
if (CONCAT_OP)
  add_definitions(-DCONCAT_OP)
endif()
if (CONV_OP)
  add_definitions(-DCONV_OP)
endif()
if (DEPTHWISECONV_OP)
  add_definitions(-DDEPTHWISECONV_OP)
endif()
if (ELEMENTWISEADD_OP)
  add_definitions(-DELEMENTWISEADD_OP)
endif()
if (FUSION_CONVADD_OP)
  add_definitions(-DFUSION_CONVADD_OP)
endif()
if (CONVADDRELU_OP)
  add_definitions(-DCONVADDRELU_OP)
endif()
if (FUSION_FC_OP)
  add_definitions(-DFUSION_FC_OP)
endif()
if (LRN_OP)
  add_definitions(-DLRN_OP)
endif()
if (MUL_OP)
  add_definitions(-DMUL_OP)
endif()
if (MULTICLASSNMS_OP)
  add_definitions(-DMULTICLASSNMS_OP)
endif()
if (POOL_OP)
  add_definitions(-DPOOL_OP)
endif()
if (PRIORBOX_OP)
  add_definitions(-DPRIORBOX_OP)
endif()
if (RELU_OP)
  add_definitions(-DRELU_OP)
endif()
if (RESHAPE_OP)
  add_definitions(-DRESHAPE_OP)
endif()
if (SIGMOID_OP)
  add_definitions(-DSIGMOID_OP)
endif()
if (SOFTMAX_OP)
  add_definitions(-DSOFTMAX_OP)
endif()
if (TRANSPOSE_OP)
  add_definitions(-DTRANSPOSE_OP)
endif()
if (FUSION_CONVADD_RELU_OP)
  add_definitions(-DFUSION_CONVADD_RELU_OP)
endif()
if (FUSION_CONVADDBNRELU_OP)
  add_definitions(-DFUSION_CONVADDBNRELU_OP)
endif()
