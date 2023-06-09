#ifndef E223D02A_48D3_432D_8CB0_D3733E4C7771
#define E223D02A_48D3_432D_8CB0_D3733E4C7771

#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>

#include "toml++/toml.h"

class YOLOv3Impl : public torch::nn::Module
{
 public:
  YOLOv3Impl(toml::node_view<toml::node> config);
  ~YOLOv3Impl() = default;

  torch::Tensor forward(torch::Tensor);

 private:
  torch::nn::Sequential m_module_list{};
  int64_t m_last_output{ 0 };
};

TORCH_MODULE(YOLOv3);

#endif /* E223D02A_48D3_432D_8CB0_D3733E4C7771 */
