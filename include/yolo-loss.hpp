#ifndef C98AFE68_11C4_4686_B6ED_169530FA0165
#define C98AFE68_11C4_4686_B6ED_169530FA0165
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/loss.h>
#include <torch/serialize/input-archive.h>

#include <ostream>
class YOLOLossImpl : public torch::nn::Module
{
 public:
  torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
  {
    // std::cout << "input predictions " << predictions.sizes() << std::endl;

    auto class_labels = targets.slice(3, torch::indexing::None, 20, 1);
    auto class_predictions = predictions.slice(3, torch::indexing::None, 20, 1);
    auto class_loss = m_cross_entropy_loss(class_predictions, class_labels);
    // std::cout << "class_labels " << class_labels.sizes() << std::endl;
    // std::cout << "class_predictions " << class_predictions.sizes() << std::endl;
    // std::cout << "class_loss " << class_loss << std::endl;

    // std::cout << std::endl;

    auto box_labels = targets.slice(3, 20, 24, 1);
    auto box_predictions = predictions.slice(3, 20, 24, 1);
    auto box_loss = m_mse_loss(box_predictions, box_labels);
    // std::cout << "box_labels " << box_labels.sizes() << std::endl;
    // std::cout << "box_predictions " << box_predictions.sizes() << std::endl;
    // std::cout << "box_loss " << box_loss << std::endl;

    // std::cout << std::endl;

    auto bojectness_labels = targets.slice(3, 24, torch::indexing::None, 1);
    auto bojectness_predictions = predictions.slice(3, 24, torch::indexing::None, 1);
    auto object_loss = m_bce(bojectness_predictions, bojectness_labels);
    // std::cout << "bojectness_labels " << bojectness_labels.sizes() << std::endl;
    // std::cout << "bojectness_predictions " << bojectness_predictions.sizes() << std::endl;
    // std::cout << "object_loss " << object_loss << std::endl;

    std::cout << std::endl;

    // std::cout << "class_loss " << class_loss.item<float>() << " box_loss " << box_loss.item<float>() << " object_loss " << object_loss.item<float>() << std::endl;
    
    return class_loss + box_loss * 10 + object_loss * 20;
  }

 private:
  /**
  @brief For binary classification error.
  */
  torch::nn::CrossEntropyLoss m_cross_entropy_loss;

  /**
  @brief For regrestion error, for continuos values like x,y,w,h.
  */
  torch::nn::MSELoss m_mse_loss;

  /**
  @brief For objectness loss.
  */
  torch::nn::BCEWithLogitsLoss m_bce;
};

TORCH_MODULE(YOLOLoss);
#endif /* C98AFE68_11C4_4686_B6ED_169530FA0165 */
