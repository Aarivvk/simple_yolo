

#include "yolo-model.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/util/variant.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/pooling.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#define INTCAST(value) (static_cast<int64_t>(std::floor(value)))

class CNNImpl : public torch::nn::Module
{
 public:
  CNNImpl(
      int64_t input_channel,
      int64_t output_channel,
      int64_t kernel_size,
      int64_t padding = 0,
      int64_t stride = 1,
      bool add_bn = true)
      : m_cnn_2d{ torch::nn::Conv2dOptions({ input_channel, output_channel, kernel_size }).padding(padding).stride(stride) },
        m_batch_norm_2d{ torch::nn::BatchNorm2dOptions(output_channel) },
        add_bn{ add_bn },
        activation{ torch::nn::LeakyReLUOptions().negative_slope(0.1) }
  {
    register_module("conv", m_cnn_2d);
    register_module("batch_norm", m_batch_norm_2d);
    register_module("Activation", activation);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    torch::Tensor out = m_cnn_2d->forward(x);
    if (add_bn)
    {
      out = m_batch_norm_2d(out);
      out = activation(out);
    }
    return out;
  }

 private:
  torch::nn::Conv2d m_cnn_2d;
  torch::nn::BatchNorm2d m_batch_norm_2d;
  bool add_bn{ true };
  torch::nn::LeakyReLU activation;
};

TORCH_MODULE(CNN);

class YOLOBlockImpl : public torch::nn::Module
{
 public:
  YOLOBlockImpl(int64_t input_channel, int64_t output_channel, bool short_cut = true)
      : m_cnn_1{ input_channel, output_channel, 1 },
        m_cnn_2{ output_channel, input_channel, 3, 1 },
        add_short_cut(short_cut)
  {
    register_module("Yolo_block_cnn_1", m_cnn_1);
    register_module("Yolo_block_cnn_2", m_cnn_2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    torch::Tensor out = m_cnn_1(x);
    out = m_cnn_2(out);
    if (add_short_cut)
    {
      out = x + out;
    }
    return out;
  }

 private:
  CNN m_cnn_1, m_cnn_2;
  bool add_short_cut;
};

TORCH_MODULE(YOLOBlock);

class YOLOPredictionImpl : public torch::nn::Module
{
 public:
  YOLOPredictionImpl(int64_t input_channel, int64_t num_classes, int64_t number_anchors)
      : m_num_classes{ num_classes },
      m_number_anchors{number_anchors},
        m_cnn_1{ input_channel, (num_classes + 5 /*x,y,w,h,obj*/), 1, 0, 1, false },
        m_activation{torch::nn::Sigmoid()}
  {
    register_module("Yolo_prediction", m_cnn_1);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    torch::Tensor out = m_activation(m_cnn_1(x));
    /*  The prediction part do not have fully connected layers.
        Prediction will be in 3D convalution network.
    */
    out = out.reshape({ x.size(0), (m_num_classes + 5), x.size(2), x.size(3) }).permute({ 0, 2, 3, 1});
    return out;
  }

 private:
  CNN m_cnn_1;
  torch::nn::Sigmoid m_activation;
  int64_t m_num_classes, m_number_anchors;
};

TORCH_MODULE(YOLOPrediction);

YOLOv3Impl::YOLOv3Impl(toml::node_view<toml::node> model_config)
{
  int64_t input_channel = model_config["input_channel"].value<int64_t>().value();
  int64_t num_classes  = model_config["num_classes"].value<int64_t>().value();
  int64_t num_anchors = model_config["num_classes"].value<int64_t>().value();
  auto layers_config = model_config["layers_structure"];
  auto num_layers = layers_config.as_array()->size();
  for (size_t i = 0; i < num_layers; i++)
  {
    auto placeholder = layers_config[i];
    int64_t filters = placeholder[0].value<int64_t>().value();
    int64_t ksize = placeholder[1].value<int64_t>().value();
    int64_t stride = placeholder[2].value<int64_t>().value();
    int64_t yb = placeholder[3].value<int64_t>().value();
    if (0 < yb)
    {
      m_module_list->push_back(CNN(m_last_output, filters, ksize, 1, stride));
      auto yb_output_filters = m_last_output;
      auto yb_input_filters = filters;
      for (size_t j = 0; j < yb; j++)
      {
        m_module_list->push_back(YOLOBlock(yb_input_filters, yb_output_filters));
        m_last_output = yb_input_filters;
        yb_output_filters = INTCAST(yb_input_filters / 2);
        yb_input_filters = m_last_output;
      }
    }
    else if (0 == yb)
    {
      if (0 == i)
      {
        m_module_list->push_back(CNN(input_channel, filters, ksize, 1, stride));
      }
      else
      {
        m_module_list->push_back(CNN(m_last_output, filters, ksize, 1, stride));
      }
      m_last_output = filters;
    }
    else
    {
      std::cerr << "Invalid block repeter " << yb << std::endl;
      exit(2);
    }
  }

  m_module_list->push_back(YOLOPrediction(m_last_output, num_classes, num_anchors));

  register_module("Yolo_module_v3", m_module_list);
}

torch::Tensor YOLOv3Impl::forward(torch::Tensor x)
{
  x = m_module_list->forward(x);
  return x;
}
