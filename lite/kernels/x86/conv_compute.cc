// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/x86/conv_compute.h"
#include <utility>
#include "lite/backends/x86/cpu_info.h"
#include "lite/kernels/x86/conv_1x1.h"
#include "lite/kernels/x86/conv_avx2.h"
#include "lite/kernels/x86/conv_avx2_group.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
void Conv2dCompute<T>::PrepareForRun() {
  auto& param = *param_.get_mutable<operators::ConvParam>();
  auto in_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  VLOG(3) << "in_dims=" << in_dims.repr();
  VLOG(3) << "w_dims=" << w_dims.repr();

  bool use_avx2 = MayIUse(lite::x86::cpu_isa_t::avx2);
  bool use_avx512 = MayIUse(lite::x86::cpu_isa_t::avx512f);

  VLOG(3) << "use_avx2=" << use_avx2 << "; use_avx512=" << use_avx512;

  int groups = param.groups;
  int ic = w_dims[1] * groups;
  int oc = w_dims[0];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];

  VLOG(3) << "groups=" << groups;
  VLOG(3) << "ic=" << ic << "; oc=" << oc << "; kh=" << kh << "; kw=" << kw;

  int pad_h = paddings[0];
  int pad_w = paddings[1];

  VLOG(3) << "pad_h=" << pad_h << "; pad_w=" << pad_w;

  int stride_h = param.strides[0];
  int stride_w = param.strides[1];

  VLOG(3) << "stride_h=" << stride_h << "; stride_w=" << stride_w;

  int dilation_h = dilations[0];
  int dilation_w = dilations[1];

  VLOG(3) << "dilation_h=" << dilation_h << "; dilation_w=" << dilation_w;

  int ih = in_dims[2];
  int iw = in_dims[3];
  int in = in_dims[0];

  VLOG(3) << "ih=" << ih << "; iw=" << iw << "; in=" << in;

  bool conv_1x1_flag = (kh == 1 && kw == 1) && (pad_h == 0 && pad_w == 0) &&
                       (stride_h == 1 && stride_w == 1) && groups == 1;
  // bool is_winorgrad = (kh == 3 && kw == 3) && (stride_h == 1 && stride_w ==
  // 1) && (dilation_h == 1 && dilation_w == 1) && group == 1;

  VLOG(3) << "conv_1x1_flag=" << conv_1x1_flag;

  if (conv_1x1_flag) {
    impl_ = new Conv1X1<PRECISION(kFloat)>;
    VLOG(3) << "invoking Conv1X1";
  } else if (use_avx2 && groups == 1 && pad_w <= 3) {
    impl_ = new AVX2Conv<PRECISION(kFloat)>;
    VLOG(3) << "invoking AVX2Conv";
  } else if (use_avx2 && groups != 1 && pad_w <= 3) {
    impl_ = new AVX2GroupConv<PRECISION(kFloat)>;
    VLOG(3) << "invoking AVX2GroupConv";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::Conv2dCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::Conv2dCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
